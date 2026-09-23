"""Orchestrator.

    User Doc
       |
    parse / OCR  ------------------ whole-doc fallback
       |
    chunk + index (RAG)
       |
       +--> extract_clauses ---> grounding verifier (deterministic)
       |                              |
       +--> segment_summary <---------+
                |
            validate (agentic, playbook + rulebook)
                |
             judge --- revise? ---> re-run with feedback (bounded)
                |
             accept
                |
        human validation packet
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional

from . import agents, grounding, ingest, playbook as playbook_mod
from .config import SETTINGS
from .llm import METER
from .retrieval import Index, chunk_document
from .schemas import PipelineResult

log = logging.getLogger("summarizer.pipeline")

Progress = Callable[[str], None]


def _noop(_: str) -> None:
    pass


def run(
    document_path: str | Path,
    playbook_path: str | Path,
    *,
    force_whole_doc: bool = False,
    allow_ocr: bool = True,
    verbose: bool = False,
    on_progress: Optional[Progress] = None,
) -> PipelineResult:
    say = on_progress or _noop

    say("1/7 parsing document")
    doc = ingest.load(document_path, allow_ocr=allow_ocr, force_whole_doc=force_whole_doc)
    say(f"    mode={doc.mode}, pages={len(doc.pages) or 'n/a'}")

    say("2/7 indexing")
    doc_index = Index(chunk_document(doc)) if doc.pages else Index([])
    rules, pb_index = playbook_mod.load(playbook_path)
    say(f"    {len(doc_index.chunks)} chunks, {len(rules)} playbook rules")

    feedback: List[str] = []
    attempts = 0
    result = None

    while attempts <= SETTINGS.max_repairs:
        attempts += 1
        say(f"--- attempt {attempts} ---")

        say("3/7 extracting clauses (grounding arm)")
        clauses = agents.extract_clauses(doc, doc_index, feedback=feedback)
        say(f"    {len(clauses.clauses)} clauses")

        say("4/7 verifying quoted spans against source")
        clauses, rate, ungrounded = grounding.verify_clauses(clauses, doc)
        note = grounding.grounding_report(clauses, rate, ungrounded)
        say(f"    grounding rate {rate:.0%}, {len(ungrounded)} unsupported clause(s)")

        say("5/7 building structured summary (segmentation arm)")
        summary = agents.segment_summary(doc, clauses, note, feedback=feedback)

        say("6/7 validating against playbook (agentic)")
        validation = agents.validate(
            doc_index, pb_index, clauses, summary, rules, feedback=feedback, verbose=verbose
        )
        say(f"    {len(validation.findings)} findings, overall={validation.overall}")

        say("7/7 judging")
        verdict = agents.judge(doc_index, clauses, summary, validation, note)
        say(f"    score={verdict.score:.2f} grounded={verdict.grounded} verdict={verdict.verdict}")

        result = PipelineResult(
            source=str(document_path),
            ingest_mode=doc.mode,
            clauses=clauses,
            summary=summary,
            validation=validation,
            judge=verdict,
            attempts=attempts,
            grounding_rate=rate,
            needs_human_review=True,  # always - the human gate is not optional
            usd_cost=round(METER.usd, 4),
        )

        if verdict.verdict == "accept" and verdict.score >= SETTINGS.judge_threshold:
            break

        feedback = [f"[{i.stage}/{i.severity.value}] {i.detail} -> {i.fix_instruction}" for i in verdict.issues]
        if not feedback:
            feedback = [f"Judge scored {verdict.score:.2f}, below the {SETTINGS.judge_threshold} bar. Be more thorough."]
        say(f"    judge requested revision; {len(feedback)} instruction(s) fed back")

    assert result is not None
    return result


def render_review_packet(result: PipelineResult) -> str:
    """Markdown handed to the human validator - the last box in the flow."""
    out: List[str] = []
    s, j, v = result.summary, result.judge, result.validation

    out.append(f"# Review packet - {s.title}\n")
    out.append(
        f"> Source `{Path(result.source).name}` | ingest `{result.ingest_mode}` | attempts {result.attempts} "
        f"| grounding {result.grounding_rate:.0%} | judge {j.score:.2f} ({j.verdict}) | ${result.usd_cost:.3f}\n"
    )
    if j.score < SETTINGS.judge_threshold:
        out.append("**The judge did not clear this analysis. Treat every statement as unverified.**\n")

    out.append(f"**Parties:** {', '.join(s.parties)}  ")
    out.append(f"**Type:** {s.agreement_type}  ")
    out.append(f"**Effective:** {s.effective_date or '-'}  |  **Term:** {s.term or '-'}  ")
    out.append(f"**Governing law:** {s.governing_law or '-'}\n")
    out.append(f"## Executive summary\n\n{s.executive_summary}\n")

    for section in s.sections:
        out.append(f"### {section.heading}")
        out += [f"- {b}" for b in section.bullets]
        if section.clause_ids:
            out.append(f"\n<sub>from: {', '.join(section.clause_ids)}</sub>")
        out.append("")

    out.append("## Playbook findings\n")
    out.append("| Rule | Status | Severity | Explanation |")
    out.append("|---|---|---|---|")
    for f in sorted(v.findings, key=lambda f: ("low", "medium", "high").index(f.severity.value), reverse=True):
        note = f.explanation.replace("|", "\\|")
        out.append(f"| {f.rule_id} {f.rule_title} | {f.status.value} | {f.severity.value} | {note} |")
    out.append(f"\n**Overall: {v.overall}** ({v.blocking_count} blocking)\n")

    redlines = [f for f in v.findings if f.suggested_redline]
    if redlines:
        out.append("## Suggested redlines\n")
        for f in redlines:
            out.append(f"- **{f.rule_id}**: {f.suggested_redline}")
        out.append("")

    out.append("## Clause evidence\n")
    for c in result.clauses.clauses:
        flag = "" if any(sp.verified for sp in c.spans) else " :warning: UNVERIFIED"
        out.append(f"### {c.heading} - {c.category} ({c.risk.value} risk){flag}")
        out.append(f"{c.summary}\n")
        for sp in c.spans:
            mark = {"exact": "OK", "fuzzy": "~", "missing": "!!"}[sp.match]
            out.append(f"> [{mark} p.{sp.page}] {sp.quote}")
        out.append("")

    if s.open_questions:
        out.append("## Open questions for the reviewer\n")
        out += [f"- [ ] {q}" for q in s.open_questions]
        out.append("")

    if j.issues:
        out.append("## Judge's outstanding concerns\n")
        out += [f"- **{i.stage}** ({i.severity.value}): {i.detail}" for i in j.issues]
        out.append("")

    out.append("---\n## Human validation\n")
    out.append("- [ ] Parties, dates and amounts checked against the document")
    out.append("- [ ] Every high-risk clause read in full")
    out.append("- [ ] Unverified quotes above resolved")
    out.append("- [ ] Findings accepted / overridden\n")
    out.append("Reviewer: ____________________   Date: ____________")
    return "\n".join(out)
