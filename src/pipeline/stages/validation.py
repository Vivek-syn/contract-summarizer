"""Validation against a playbook - rule-major, not page-wise.

Validation is a rule x clause matrix. A rule such as "liability capped at 12
months' fees" needs the relevant clauses wherever they live; validating page 7's
clauses in isolation asks a question that has no answer, and then forces a
consolidation step to reconcile N partial verdicts per rule.

So: one call per rule, with the clauses relevant to that rule retrieved for it.
Small calls, trivially parallel, one clean finding each, and no consolidation
step at all because rules do not overlap.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

from src.core.logging.structured import get_logger
from src.schemas.documents import (
    AnalysisResult,
    Clause,
    Finding,
    FindingStatus,
    Risk,
    ValidationResult,
)
from src.services.llm.adapter import LLMAdapter
from src.utils.retrieval import Index, index_clauses

logger = get_logger(__name__)

SYSTEM = """You are a contract reviewer checking ONE rule from your organisation's
playbook against a document.

You are given the rule, and the clauses from the document most relevant to it.

Decide one of:
- `compliant`  - the document satisfies the rule.
- `deviation`  - the document addresses this but departs from the rule.
- `missing`    - the document is silent where the rule requires language.
- `unclear`    - the relevant text is ambiguous or was not legible.

Rules for your answer:
- Cite the clause ids you relied on. If you cite none, say why in the explanation.
- `missing` is a real and common answer. Do not stretch an unrelated clause to
  make something look compliant.
- Severity is the rule's severity, raised only if this document's specific
  wording makes it worse than the rule anticipated.
- Suggest a redline only for `deviation` or `missing`, and make it text that
  could be pasted into the contract."""


def _rule_query(rule: Dict[str, str]) -> str:
    """Retrieval query for a rule: its title, text and fallback together."""
    return " ".join(
        str(rule.get(field, "")) for field in ("title", "rule", "fallback", "preferred_language")
    )


async def _validate_one(
    rule: Dict[str, str],
    clauses: List[Clause],
    adapter: LLMAdapter,
    *,
    model: Optional[str] = None,
) -> Finding:
    """Validate a single rule against its relevant clauses."""
    if clauses:
        rendered = "\n\n".join(
            f"<clause id=\"{c.id}\" page=\"{c.page}\" risk=\"{c.risk.value}\">\n"
            f"{c.heading} ({c.category})\n{c.summary}\n\nQuoted text: {c.quote}\n</clause>"
            for c in clauses
        )
    else:
        rendered = "(no clause in this document matched this rule)"

    rule_text = "\n".join(
        f"{label}: {rule[key]}"
        for label, key in (
            ("Rule", "rule"),
            ("Fallback position", "fallback"),
            ("Preferred language", "preferred_language"),
        )
        if rule.get(key)
    )

    user = (
        f"<rule id=\"{rule['id']}\" title=\"{rule.get('title', '')}\" "
        f"severity=\"{rule.get('severity', 'medium')}\">\n{rule_text}\n</rule>\n\n"
        f"<relevant_clauses>\n{rendered}\n</relevant_clauses>\n\n"
        f"Assess rule {rule['id']} against this document."
    )

    finding = await adapter.structured(
        Finding,
        system=SYSTEM,
        user=user,
        tool_name="record_finding",
        tool_description=f"Record the outcome for playbook rule {rule['id']}.",
        model=model,
        max_tokens=8000,
    )
    # The model does not get to rename the rule it was asked about.
    finding.rule_id = rule["id"]
    finding.rule_title = rule.get("title", finding.rule_title)
    return finding


async def run(
    analysis: AnalysisResult,
    rules: List[Dict[str, str]],
    adapter: LLMAdapter,
    *,
    concurrency: int = 4,
    top_k: int = 6,
    model: Optional[str] = None,
) -> ValidationResult:
    """Validate every playbook rule, in parallel.

    Args:
        analysis: The extracted clauses.
        rules: Playbook rules, each with at least ``id``, ``title``, ``rule``.
        adapter: LLM adapter.
        concurrency: Rules validated at once.
        top_k: Clauses retrieved per rule.
        model: Model override.

    Returns:
        A :class:`ValidationResult`. A rule whose call failed becomes an
        ``unclear`` finding rather than vanishing - a silently absent rule is
        indistinguishable from a passing one, which would be dangerous.
    """
    if not rules:
        return ValidationResult(findings=[], overall="pass")

    index: Index = index_clauses(analysis.clauses)
    by_id = {c.id: c for c in analysis.clauses}
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _guarded(rule: Dict[str, str]) -> Finding:
        async with semaphore:
            hits = index.search(_rule_query(rule), top_k=top_k)
            relevant = [by_id[h.id] for h in hits if h.id in by_id]
            return await _validate_one(rule, relevant, adapter, model=model)

    logger.info("validation: %d rules, %d concurrent", len(rules), concurrency)
    outcomes = await asyncio.gather(*(_guarded(r) for r in rules), return_exceptions=True)

    findings: List[Finding] = []
    for rule, outcome in zip(rules, outcomes):
        if isinstance(outcome, BaseException):
            logger.warning("rule %s failed to validate: %s", rule["id"], outcome)
            findings.append(
                Finding(
                    rule_id=rule["id"],
                    rule_title=rule.get("title", rule["id"]),
                    status=FindingStatus.unclear,
                    severity=Risk(rule.get("severity", "medium")),
                    explanation=f"This rule could not be checked: {outcome}",
                )
            )
        else:
            findings.append(outcome)

    blocking = sum(
        1
        for f in findings
        if f.status in (FindingStatus.deviation, FindingStatus.missing) and f.severity == Risk.high
    )
    if blocking:
        overall = "fail"
    elif any(f.status != FindingStatus.compliant for f in findings):
        overall = "pass_with_comments"
    else:
        overall = "pass"

    logger.info("validation: %d findings, %d blocking, overall=%s", len(findings), blocking, overall)
    return ValidationResult(findings=findings, blocking_count=blocking, overall=overall)
