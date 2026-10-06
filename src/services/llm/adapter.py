"""LiteLLM adapter with per-call usage accounting.

Three provider facts are baked in here, each verified against LiteLLM's own
transforms rather than assumed:

1. ``budget_tokens`` is rejected on Anthropic's current models - adaptive
   thinking is the only mode.
2. ``output_config`` must be a top-level kwarg for Anthropic. Passing it as
   ``extra_body`` sends a literal ``extra_body`` key, which Anthropic rejects;
   and for OpenAI it is not dropped at all but tucked into ``extra_body``, which
   OpenAI then rejects. So it is Anthropic-only.
3. ``reasoning_effort`` sent to Anthropic is rewritten to the deprecated
   ``thinking: {type: "enabled", budget_tokens: N}``, which is a 400. So it is
   OpenAI-only.

Structured output uses a forced tool call on OpenAI (which supports it) and
``auto`` plus an explicit instruction on Anthropic (whose newest models reject
forced tool choice, and where it conflicts with extended thinking anyway).
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar

import litellm
from pydantic import BaseModel, ValidationError

from src.core.config.settings import get_settings
from src.core.errors.exceptions import LLMError, SchemaRepairExhausted
from src.core.logging.structured import get_logger
from src.schemas.state import Usage

logger = get_logger(__name__)

litellm.drop_params = True

T = TypeVar("T", bound=BaseModel)

_OPTIONAL_PARAMS = ("thinking", "output_config", "reasoning_effort", "tool_choice")
_OPENAI_EFFORT = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}
_OPENAI_LIKE = {"openai", "azure", "openrouter"}


def provider_of(model: str) -> str:
    """Resolve the LiteLLM provider for a model string."""
    try:
        _, provider, _, _ = litellm.get_llm_provider(model=model)
        return provider or ""
    except Exception:
        return model.split("/", 1)[0] if "/" in model else ""


def missing_keys(model: str) -> List[str]:
    """Env vars this model needs but does not have."""
    try:
        return list(litellm.validate_environment(model=model).get("missing_keys") or [])
    except Exception:
        return []


class UsageTracker:
    """Accumulates tokens and cost. Thread-safe; stages may run concurrently."""

    def __init__(self) -> None:
        self._usage = Usage()
        self._lock = threading.Lock()

    def record(self, response: Any) -> Usage:
        """Record one response and return its own usage delta."""
        delta = Usage(calls=1)
        raw = getattr(response, "usage", None)
        if raw is not None:
            delta.input_tokens = int(getattr(raw, "prompt_tokens", 0) or 0)
            delta.output_tokens = int(getattr(raw, "completion_tokens", 0) or 0)
        try:
            delta.cost_usd = float(litellm.completion_cost(completion_response=response) or 0.0)
        except Exception:
            delta.cost_usd = 0.0  # a pricing-table miss must not fail a run
        with self._lock:
            self._usage.add(delta)
        return delta

    @property
    def usage(self) -> Usage:
        with self._lock:
            return self._usage.model_copy(deep=True)

    def reset(self) -> None:
        with self._lock:
            self._usage = Usage()


class LLMAdapter:
    """Wraps LiteLLM so call sites never touch provider quirks."""

    def __init__(self, tracker: Optional[UsageTracker] = None) -> None:
        self.tracker = tracker or UsageTracker()
        self.settings = get_settings().llm

    def _base_kwargs(self, model: str, *, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        provider = provider_of(model)
        budget = max_tokens or self.settings.max_tokens

        # On OpenAI reasoning models the cap is max_completion_tokens, which
        # INCLUDES reasoning tokens - a budget sized for visible output gets
        # eaten by thinking and the turn ends with nothing emitted.
        if provider in _OPENAI_LIKE:
            try:
                if litellm.supports_reasoning(model=model):
                    budget = max(budget, budget + 16000)
            except Exception:
                pass

        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": budget,
            "num_retries": self.settings.num_retries,
            "timeout": self.settings.request_timeout,
        }
        if provider == "anthropic":
            if self.settings.thinking:
                kwargs["thinking"] = {"type": "adaptive"}
            if self.settings.effort:
                kwargs["output_config"] = {"effort": self.settings.effort}
        elif provider in _OPENAI_LIKE:
            if self.settings.effort:
                kwargs["reasoning_effort"] = _OPENAI_EFFORT.get(self.settings.effort, "medium")
        return kwargs

    @staticmethod
    def _prepare_tools(tools: Any, provider: str) -> Any:
        """Strip ``strict`` for providers that enforce it.

        OpenAI's strict mode requires every property in ``required``; our schemas
        have genuinely optional fields. Anthropic ignores the flag entirely.
        """
        if not tools or provider not in {"openai", "azure"}:
            return tools
        cleaned = []
        for tool in tools:
            copy = dict(tool)
            copy["function"] = {k: v for k, v in tool["function"].items() if k != "strict"}
            cleaned.append(copy)
        return cleaned

    async def complete(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        **overrides: Any,
    ) -> Any:
        """One completion, degrading gracefully on unsupported params."""
        resolved = model or self.settings.model
        kwargs = self._base_kwargs(resolved, max_tokens=overrides.pop("max_tokens", None))
        kwargs.update(overrides)
        if "tools" in kwargs:
            kwargs["tools"] = self._prepare_tools(kwargs["tools"], provider_of(resolved))

        attempt = dict(kwargs)
        for _ in range(len(_OPTIONAL_PARAMS) + 1):
            try:
                response = await litellm.acompletion(messages=messages, **attempt)
                self.tracker.record(response)
                return response
            except litellm.BadRequestError as exc:
                dropped = next((p for p in _OPTIONAL_PARAMS if p in attempt), None)
                if dropped is None:
                    raise LLMError(f"model rejected the request: {exc}") from exc
                logger.warning("400 from provider (%s); retrying without %r", str(exc)[:160], dropped)
                attempt.pop(dropped)
            except Exception as exc:
                raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        raise LLMError("exhausted parameter-degradation retries")

    async def structured(
        self,
        schema: Type[T],
        *,
        system: str,
        user: Any,
        tool_name: str,
        tool_description: str,
        model: Optional[str] = None,
        max_repairs: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> T:
        """Return a validated pydantic object, repairing from validation errors."""
        resolved = model or self.settings.model
        repairs = self.settings.max_repairs if max_repairs is None else max_repairs
        tool = tool_from_model(schema, tool_name, tool_description)

        provider = provider_of(resolved)
        if provider in {"openai", "azure"}:
            choice: Any = {"type": "function", "function": {"name": tool_name}}
        else:
            choice = "auto"

        instruction = (
            f"\n\nRespond by calling the `{tool_name}` tool exactly once. Do not reply with prose."
        )
        content = (
            list(user) + [{"type": "text", "text": instruction}]
            if isinstance(user, list)
            else f"{user}{instruction}"
        )
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]

        last_error = ""
        for attempt in range(repairs + 1):
            response = await self.complete(
                messages, model=resolved, tools=[tool], tool_choice=choice, max_tokens=max_tokens
            )
            choice_obj = response.choices[0]
            message = choice_obj.message
            finish = getattr(choice_obj, "finish_reason", None)
            text = (message.content or "").strip()
            payload = _first_tool_call(response, tool_name)

            if payload is not None:
                try:
                    return schema.model_validate(payload)
                except ValidationError as exc:
                    last_error = f"Your tool arguments failed validation:\n{exc}"
            elif finish == "length":
                raise SchemaRepairExhausted(
                    f"{tool_name}: hit the output cap before producing a tool call "
                    f"(finish_reason=length). Raise LLM_MAX_TOKENS or lower LLM_EFFORT."
                )
            else:
                last_error = f"You replied with text instead of calling `{tool_name}`."

            logger.warning(
                "structured(%s) attempt %d failed: finish_reason=%s text=%r",
                tool_name, attempt + 1, finish, text[:300] or "(empty)",
            )
            messages.append({"role": "assistant", "content": text or "(no tool call)"})
            messages.append({"role": "user", "content": last_error + instruction})

        raise SchemaRepairExhausted(
            f"{tool_name}: no schema-valid tool call after {repairs + 1} attempts: {last_error}"
        )

    async def structured_many(
        self,
        calls: Sequence[Dict[str, Any]],
        *,
        concurrency: int = 4,
    ) -> List[Any]:
        """Run several ``structured`` calls concurrently.

        Returns results positionally; a failed call yields its exception rather
        than cancelling its siblings, so one bad rule does not lose the rest.
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def _run(kwargs: Dict[str, Any]) -> Any:
            async with semaphore:
                return await self.structured(**kwargs)

        return await asyncio.gather(*(_run(dict(c)) for c in calls), return_exceptions=True)


def tool_from_model(schema: Type[BaseModel], name: str, description: str) -> Dict[str, Any]:
    """Build a strict OpenAI-format tool definition from a pydantic model."""
    params = schema.model_json_schema()
    params["additionalProperties"] = False
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": params,
            "strict": True,
        },
    }


def _first_tool_call(response: Any, name: str) -> Optional[Dict[str, Any]]:
    calls = getattr(response.choices[0].message, "tool_calls", None) or []
    for call in calls:
        if call.function.name == name:
            # Never string-match serialized tool input - escaping varies by model.
            return json.loads(call.function.arguments)
    return None
