"""The four model-driven stages.

- ``extract_clauses``   -> Grounding arm: every important clause + verbatim spans
- ``segment_summary``   -> Segmentation arm: structured summary
- ``validate``          -> agentic loop against the playbook / rulebook (RAG + tools)
- ``judge``             -> LLM-as-judge gate before human review
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .config import SETTINGS
from .llm import agent_loop, structured, tool_from_model
from .retrieval import Index
from .schemas import (
    ClauseSet,
    JudgeVerdict,
    PlaybookFinding,
    StructuredSummary,
    ValidationReport,
)

log = logging.getLogger("summarizer.agents")

CLAUSE_SYSTEM = """You are a contract analyst. You extract clauses; you never invent them.

Rules:
- Every clause MUST carry at least one span whose `quote` is copied VERBATIM from
  the document, character for character. A paraphrased quote is a failure.
- Cover the whole agreement, including clauses that are unfavourable or unusual.
  Omitting a bad clause is the worst error you can make here.
- Prefer several short precise quotes over one long one.
- If something is genuinely absent (e.g. no limitation of liability), do not
  fabricate a clause for it - the validator handles missing-clause findings."""

SUMMARY_SYSTEM = """You are a contract analyst producing a structured summary for a reviewer
who has not read the document.

Rules:
- Ground every statement in the extracted clauses you are given; cite their ids.
- Segment by business meaning (Commercials, Term & Termination, Liability & Indemnity,
  IP & Data, Compliance, Other), not by the document's own numbering.
- State amounts, dates and durations exactly as written.
- Anything you could not determine goes in `open_questions` rather than being guessed."""

VALIDATOR_SYSTEM = """You are a contract reviewer checking a document against your
organisation's playbook.

You have tools. Use them - do not answer from the summary alone:
- `search_document` for the contract's own wording
- `search_playbook` for the governing rule
- `get_clause` to pull a previously extracted clause in full

Work rule by rule. For each playbook rule decide: compliant, deviation, missing, or
unclear, and record it with `record_finding`. `missing` means the contract is silent
where the playbook requires language. When you have covered every rule, stop calling
tools and say DONE."""

JUDGE_SYSTEM = """You are an adversarial reviewer grading another model's contract analysis.
You are not being helpful to the analyst - you are protecting the human who will sign this.

Grade on:
1. Grounding - does every material claim trace to a VERIFIED quote? The grounding
   report tells you which spans could not be located; unverified spans are hallucinations
   until proven otherwise.
2. Coverage - were important clauses missed? Check the retrieved passages for clause
   types absent from the extraction.
3. Fidelity - are amounts, dates, durations and party names exactly right?
4. Playbook - are the findings defensible given the rules quoted?

Score conservatively. Anything a reviewer would be embarrassed to send onward is a revise.
Every issue must carry a concrete `fix_instruction` the next attempt can act on."""


def _feedback_block(feedback: Optional[List[str]]) -> str:
    if not feedback:
        return ""
    joined = "\n".join(f"- {item}" for item in feedback)
    return f"\n\n<judge_feedback>\nA previous attempt was rejected. Fix these:\n{joined}\n</judge_feedback>"


def extract_clauses(doc, index: Index, *, feedback: Optional[List[str]] = None) -> ClauseSet:
    """Grounding arm: highlight all important clauses with verbatim spans."""
    context = doc.as_message_content()
    hints = index.render(index.search("indemnity liability termination payment confidentiality governing law", top_k=6))
    prompt: List[Dict[str, Any]] = list(context) + [
        {
            "type": "text",
            "text": (
                "Extract every important clause from this agreement.\n\n"
                f"<passages_of_interest>\n{hints}\n</passages_of_interest>"
                + _feedback_block(feedback)
            ),
        }
    ]
    return structured(
        ClauseSet,
        system=CLAUSE_SYSTEM,
        user=prompt,
        tool_name="record_clauses",
        tool_description="Record every important clause with verbatim supporting spans.",
    )


def segment_summary(
    doc, clauses: ClauseSet, grounding_note: str, *, feedback: Optional[List[str]] = None
) -> StructuredSummary:
    """Segmentation arm: structured summary built on the verified clause set."""
    payload = clauses.model_dump_json(indent=2)
    prompt = (
        f"<clauses>\n{payload}\n</clauses>\n\n"
        f"<grounding_report>\n{grounding_note}\n</grounding_report>\n\n"
        "Produce the structured summary. Do not rely on any clause whose spans are "
        "unverified; if you must mention one, flag the uncertainty in `open_questions`."
        + _feedback_block(feedback)
    )
    return structured(
        StructuredSummary,
        system=SUMMARY_SYSTEM,
        user=prompt,
        tool_name="record_summary",
        tool_description="Record the structured summary of the agreement.",
    )


def validate(
    doc_index: Index,
    playbook_index: Index,
    clauses: ClauseSet,
    summary: StructuredSummary,
    playbook_rules: List[Dict[str, str]],
    *,
    feedback: Optional[List[str]] = None,
    verbose: bool = False,
) -> ValidationReport:
    """Agentic validation: the model drives retrieval until it has covered every rule."""
    by_id = {c.id: c for c in clauses.clauses}
    collected: List[PlaybookFinding] = []

    def search_document(query: str) -> str:
        return doc_index.render(doc_index.search(query))

    def search_playbook(query: str) -> str:
        return playbook_index.render(playbook_index.search(query, top_k=4))

    def get_clause(clause_id: str) -> str:
        clause = by_id.get(clause_id)
        return clause.model_dump_json(indent=2) if clause else f"No clause with id {clause_id!r}."

    def record_finding(**kwargs: Any) -> str:
        finding = PlaybookFinding.model_validate(kwargs)
        collected.append(finding)
        return f"Recorded {finding.rule_id} as {finding.status.value} ({finding.severity.value})."

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_document",
                "description": "Full-text search over the contract. Use legal terms of art as the query.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_playbook",
                "description": "Search the organisation's playbook and rulebook for the governing rule.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_clause",
                "description": "Retrieve one previously extracted clause in full by id.",
                "parameters": {
                    "type": "object",
                    "properties": {"clause_id": {"type": "string"}},
                    "required": ["clause_id"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
        tool_from_model(PlaybookFinding, "record_finding", "Record the outcome for exactly one playbook rule."),
    ]

    rule_list = "\n".join(f"- {r['id']}: {r['title']}" for r in playbook_rules)
    user = (
        f"<summary>\n{summary.model_dump_json(indent=2)}\n</summary>\n\n"
        f"<clause_ids>\n{', '.join(by_id)}\n</clause_ids>\n\n"
        f"<rules_to_check>\n{rule_list}\n</rules_to_check>\n\n"
        "Check every rule above against the contract and record a finding for each."
        + _feedback_block(feedback)
    )

    def trace(name: str, args: Dict[str, Any], result: Any) -> None:
        if verbose:
            preview = str(result).replace("\n", " ")[:90]
            log.info("  tool %s(%s) -> %s", name, json.dumps(args)[:70], preview)

    agent_loop(
        system=VALIDATOR_SYSTEM,
        user=user,
        tools=tools,
        impls={
            "search_document": search_document,
            "search_playbook": search_playbook,
            "get_clause": get_clause,
            "record_finding": record_finding,
        },
        on_step=trace,
    )

    blocking = sum(1 for f in collected if f.status.value in {"deviation", "missing"} and f.severity.value == "high")
    if blocking:
        overall = "fail"
    elif any(f.status.value != "compliant" for f in collected):
        overall = "pass_with_comments"
    else:
        overall = "pass"
    return ValidationReport(findings=collected, blocking_count=blocking, overall=overall)


def judge(
    doc_index: Index,
    clauses: ClauseSet,
    summary: StructuredSummary,
    validation: ValidationReport,
    grounding_note: str,
) -> JudgeVerdict:
    """The Judge box: a gate, not a rubber stamp."""
    sample = doc_index.render(doc_index.search("indemnity liability termination payment notice", top_k=6))
    user = (
        f"<grounding_report>\n{grounding_note}\n</grounding_report>\n\n"
        f"<clauses>\n{clauses.model_dump_json(indent=2)}\n</clauses>\n\n"
        f"<summary>\n{summary.model_dump_json(indent=2)}\n</summary>\n\n"
        f"<validation>\n{validation.model_dump_json(indent=2)}\n</validation>\n\n"
        f"<source_passages>\n{sample}\n</source_passages>\n\n"
        "Grade this analysis."
    )
    return structured(
        JudgeVerdict,
        system=JUDGE_SYSTEM,
        user=user,
        tool_name="record_verdict",
        tool_description="Record the grading verdict for this analysis.",
        model=SETTINGS.judge_model,
    )
