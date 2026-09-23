"""Thin LiteLLM layer: structured output, an agentic tool loop, and cost tracking.

Two Anthropic constraints shape this file:

1. ``budget_tokens`` is rejected on Opus 5 - adaptive thinking is the only mode.
2. Forced tool choice (``tool_choice="required"`` or a named tool) is incompatible
   with extended thinking, and is rejected outright on the newest models. So we
   use ``tool_choice="auto"`` + ``strict`` schemas + an explicit instruction
   naming the tool, and repair from the validation error on a miss.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, TypeVar

import litellm
from pydantic import BaseModel, ValidationError

from .config import SETTINGS

log = logging.getLogger("summarizer.llm")

litellm.drop_params = True  # silently drop params a provider does not accept

T = TypeVar("T", bound=BaseModel)

# Params a provider may reject depending on model/proxy; stripped one at a time on 400.
_OPTIONAL_PARAMS = ("thinking", "output_config", "reasoning_effort", "tool_choice")

# Anthropic takes low|medium|high|xhigh|max; OpenAI takes minimal|low|medium|high.
_OPENAI_EFFORT = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}


def provider_of(model: str) -> str:
    """Resolve the LiteLLM provider for a model string, e.g. 'anthropic', 'openai'."""
    try:
        _, provider, _, _ = litellm.get_llm_provider(model=model)
        return provider or ""
    except Exception:
        return model.split("/", 1)[0] if "/" in model else ""


def missing_keys(model: str) -> List[str]:
    """Env vars this model needs but does not have. Empty means good to go."""
    try:
        return list(litellm.validate_environment(model=model).get("missing_keys") or [])
    except Exception:
        return []


class CostMeter:
    def __init__(self) -> None:
        self.usd = 0.0
        self.calls = 0

    def add(self, response: Any) -> None:
        self.calls += 1
        try:
            self.usd += float(litellm.completion_cost(completion_response=response) or 0.0)
        except Exception:  # a pricing-table miss is not worth failing a run over
            pass


METER = CostMeter()


def _base_kwargs(model: str) -> Dict[str, Any]:
    """Per-provider request params.

    These cannot be shared across providers, and the failure modes are quiet:

    - ``output_config`` is not dropped for OpenAI - LiteLLM tucks it into
      ``extra_body`` and OpenAI rejects the unknown field.
    - ``reasoning_effort`` sent to Anthropic is rewritten to the deprecated
      ``thinking: {type: "enabled", budget_tokens: N}``, which Opus 5 rejects.

    So each provider gets only the knobs it actually understands.
    """
    provider = provider_of(model)

    # On OpenAI reasoning models the cap is max_completion_tokens, which INCLUDES
    # reasoning tokens - so a visible-output budget sized for a normal model gets
    # eaten by thinking and the turn ends at finish_reason=length with nothing
    # emitted. Give reasoning models headroom on top of the configured budget.
    max_out = SETTINGS.max_tokens
    try:
        if provider in {"openai", "azure"} and litellm.supports_reasoning(model=model):
            max_out = max(max_out, SETTINGS.max_tokens + 16000)
    except Exception:
        pass

    kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_out,  # LiteLLM maps this to max_completion_tokens where needed
        "num_retries": 3,
    }

    if provider == "anthropic":
        if SETTINGS.thinking:
            # Adaptive thinking: no budget_tokens, ever.
            kwargs["thinking"] = {"type": "adaptive"}
        if SETTINGS.effort:
            # Top-level kwarg on purpose: LiteLLM forwards unknown kwargs into the
            # Anthropic body, but forwards `extra_body` as a literal (invalid) key.
            kwargs["output_config"] = {"effort": SETTINGS.effort}
    elif provider in {"openai", "azure", "openrouter"}:
        if SETTINGS.effort:
            # Silently dropped by LiteLLM on non-reasoning models, which is what we want.
            kwargs["reasoning_effort"] = _OPENAI_EFFORT.get(SETTINGS.effort, "medium")
    # Any other provider gets the plain request; drop_params handles the rest.
    return kwargs


def _prepare_tools(tools: Any, provider: str) -> Any:
    """Strip ``strict`` for providers that enforce it.

    OpenAI's strict mode requires every property to appear in ``required``; our
    schemas have genuinely optional fields (``Span.char_start``,
    ``PlaybookFinding.suggested_redline``), so strict would 400. Anthropic
    ignores the flag entirely, so it is only ever a no-op or a hazard.
    """
    if not tools or provider not in {"openai", "azure"}:
        return tools
    cleaned = []
    for tool in tools:
        copy = dict(tool)
        copy["function"] = {k: v for k, v in tool["function"].items() if k != "strict"}
        cleaned.append(copy)
    return cleaned


def complete(messages: List[Dict[str, Any]], *, model: Optional[str] = None, **overrides: Any) -> Any:
    """``litellm.completion`` with graceful degradation on unsupported params."""
    resolved = model or SETTINGS.model
    kwargs = _base_kwargs(resolved)
    kwargs.update(overrides)
    if "tools" in kwargs:
        kwargs["tools"] = _prepare_tools(kwargs["tools"], provider_of(resolved))

    attempt_kwargs = dict(kwargs)
    for _ in range(len(_OPTIONAL_PARAMS) + 1):
        try:
            response = litellm.completion(messages=messages, **attempt_kwargs)
            METER.add(response)
            return response
        except litellm.BadRequestError as exc:
            dropped = next((p for p in _OPTIONAL_PARAMS if p in attempt_kwargs), None)
            if dropped is None:
                raise
            log.warning("400 from provider (%s); retrying without %r", str(exc)[:160], dropped)
            attempt_kwargs.pop(dropped)
    raise RuntimeError("exhausted parameter-degradation retries")


def tool_from_model(schema: Type[BaseModel], name: str, description: str) -> Dict[str, Any]:
    """Build a strict OpenAI-format tool definition from a pydantic model.

    Note: LiteLLM's Anthropic transform drops ``strict`` (it only forwards
    name/description/input_schema), so schema conformance is enforced by the
    repair loop in ``structured()``, not by the provider. The flag is kept so
    the same tool works unchanged if you point LiteLLM at a strict-capable route.
    """
    params = schema.model_json_schema()
    params["additionalProperties"] = False
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": params, "strict": True},
    }


def _first_tool_call(response: Any, name: str) -> Optional[Dict[str, Any]]:
    calls = getattr(response.choices[0].message, "tool_calls", None) or []
    for call in calls:
        if call.function.name == name:
            # Never string-match serialized tool input - escaping varies by model.
            return json.loads(call.function.arguments)
    return None


def structured(
    schema: Type[T],
    *,
    system: str,
    user: str | List[Dict[str, Any]],
    tool_name: str,
    tool_description: str,
    model: Optional[str] = None,
    max_repairs: int = 2,
) -> T:
    """Get a validated pydantic object back from the model.

    Uses ``tool_choice="auto"`` plus an explicit instruction, then feeds any
    validation error back as a repair turn.
    """
    tool = tool_from_model(schema, tool_name, tool_description)
    instruction = f"\n\nRespond by calling the `{tool_name}` tool exactly once. Do not reply with prose."
    if isinstance(user, list):
        content: Any = user + [{"type": "text", "text": instruction}]
    else:
        content = user + instruction

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]

    # OpenAI still supports forcing a named tool; the newest Anthropic models 400 on it.
    resolved = model or SETTINGS.model
    if provider_of(resolved) in {"openai", "azure"}:
        choice: Any = {"type": "function", "function": {"name": tool_name}}
    else:
        choice = "auto"

    last_error = ""
    for attempt in range(max_repairs + 1):
        response = complete(messages, model=model, tools=[tool], tool_choice=choice)
        message = response.choices[0].message
        payload = _first_tool_call(response, tool_name)
        finish = getattr(response.choices[0], "finish_reason", None)
        text = (message.content or "").strip()

        if payload is not None:
            try:
                return schema.model_validate(payload)
            except ValidationError as exc:
                last_error = f"Your tool arguments failed validation:\n{exc}"
        elif finish == "length":
            # Reasoning tokens count toward the output cap, so a reasoning model at
            # high effort can burn the whole budget before emitting the tool call.
            last_error = (
                f"The response hit the {SETTINGS.max_tokens}-token output cap before "
                f"producing a tool call (finish_reason=length). Raise SUMMARIZER_MAX_TOKENS "
                f"or lower SUMMARIZER_EFFORT."
            )
        else:
            last_error = f"You replied with text instead of calling `{tool_name}`. Call the tool."

        # Log what actually came back - guessing at this costs minutes per attempt.
        log.warning(
            "structured(%s) attempt %d failed: finish_reason=%s, text=%r",
            tool_name, attempt + 1, finish, text[:300] or "(empty)",
        )
        if finish == "length":
            raise RuntimeError(f"{tool_name}: {last_error}")

        messages.append({"role": "assistant", "content": text or "(no tool call)"})
        messages.append({"role": "user", "content": last_error + instruction})

    raise RuntimeError(f"{tool_name}: no schema-valid tool call after {max_repairs + 1} attempts: {last_error}")


ToolImpl = Callable[..., Any]


def agent_loop(
    *,
    system: str,
    user: str,
    tools: Sequence[Dict[str, Any]],
    impls: Dict[str, ToolImpl],
    model: Optional[str] = None,
    max_turns: Optional[int] = None,
    on_step: Optional[Callable[[str, Dict[str, Any], Any], None]] = None,
) -> List[Dict[str, Any]]:
    """Run a real tool-calling loop until the model stops asking for tools.

    Returns the full transcript so the caller can hand it to a final
    ``structured()`` call for the answer.
    """
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    turns = max_turns or SETTINGS.max_agent_turns

    for _ in range(turns):
        response = complete(messages, model=model, tools=list(tools), tool_choice="auto")
        message = response.choices[0].message
        calls = getattr(message, "tool_calls", None) or []

        messages.append(message.model_dump() if hasattr(message, "model_dump") else dict(message))
        if not calls:
            return messages

        # Parallel tool calls must all come back before the next assistant turn.
        for call in calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                args, result = {}, f"ERROR: invalid JSON arguments ({exc})"
            else:
                impl = impls.get(name)
                if impl is None:
                    result = f"ERROR: unknown tool {name!r}"
                else:
                    try:
                        result = impl(**args)
                    except Exception as exc:  # surface tool errors to the model, don't crash the run
                        result = f"ERROR: {type(exc).__name__}: {exc}"
            if on_step:
                on_step(name, args, result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": name,
                    "content": result if isinstance(result, str) else json.dumps(result, default=str),
                }
            )

    log.warning("agent_loop hit the %d-turn ceiling", turns)
    return messages
