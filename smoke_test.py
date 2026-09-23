"""Offline smoke test - exercises every deterministic stage without touching the API.

    python smoke_test.py
"""
from __future__ import annotations

from summarizer import grounding, ingest, playbook
from summarizer.pipeline import render_review_packet
from summarizer.retrieval import Index, chunk_document
from summarizer.schemas import (
    Clause,
    ClauseSet,
    JudgeVerdict,
    PipelineResult,
    PlaybookFinding,
    Span,
    StructuredSummary,
    SummarySection,
)

doc = ingest.load("samples/sample_msa.txt")
assert doc.mode == "extracted_text", doc.mode
print(f"ingest        OK  mode={doc.mode} chars={len(doc.text)}")

chunks = chunk_document(doc)
index = Index(chunks)
hits = index.search("limitation of liability cap aggregate")
assert hits and "liability" in hits[0].text.lower()
print(f"retrieval     OK  {len(chunks)} chunks, top hit p.{hits[0].page}")

rules, pb_index = playbook.load("playbook/default_playbook.yaml")
pb_hits = pb_index.search("liability cap twelve months fees")
assert pb_hits and pb_hits[0].id == "LIAB-01", pb_hits[0].id if pb_hits else None
print(f"playbook      OK  {len(rules)} rules, top rule {pb_hits[0].id}")

clauses = ClauseSet(
    clauses=[
        Clause(
            id="liability-cap-1",
            category="Limitation of Liability",
            heading="Aggregate liability cap",
            summary="Liability is capped at 3x fees over 36 months.",
            risk="high",
            rationale="Well above the 12-month cap the playbook allows.",
            # exact quote
            spans=[Span(quote="shall not exceed three (3) times the total fees paid", page=1)],
        ),
        Clause(
            id="payment-1",
            category="Payment Terms",
            heading="Net 90 payment",
            summary="Invoices payable within 90 days.",
            risk="medium",
            rationale="Deviation from net 30.",
            # near-miss quote: whitespace and casing differ -> should fuzzy-match
            spans=[Span(quote="Invoices  are payable   within NINETY (90) days of receipt", page=1)],
        ),
        Clause(
            id="hallucinated-1",
            category="Insurance",
            heading="Insurance obligation",
            summary="Supplier must carry $10m of cover.",
            risk="low",
            rationale="n/a",
            # not in the document at all -> must be flagged missing
            spans=[Span(quote="Supplier shall maintain insurance of not less than ten million dollars", page=1)],
        ),
    ]
)

clauses, rate, ungrounded = grounding.verify_clauses(clauses, doc)
marks = {c.id: c.spans[0].match for c in clauses.clauses}
assert marks["liability-cap-1"] == "exact", marks
assert marks["payment-1"] in {"exact", "fuzzy"}, marks
assert marks["hallucinated-1"] == "missing", marks
assert [c.id for c in ungrounded] == ["hallucinated-1"]
print(f"grounding     OK  rate={rate:.0%} marks={marks}")

result = PipelineResult(
    source="samples/sample_msa.txt",
    ingest_mode=doc.mode,
    clauses=clauses,
    summary=StructuredSummary(
        title="Northwind / Calder MSA",
        parties=["Northwind Analytics Ltd", "Calder Retail Group plc"],
        agreement_type="Master Services Agreement",
        effective_date="2025-03-14",
        term="24 months, auto-renewing",
        governing_law="Singapore",
        executive_summary="Supplier-unfavourable MSA with a 3x/36-month cap and net-90 terms.",
        sections=[
            SummarySection(
                heading="Liability & Indemnity",
                bullets=["Cap is 3x fees over 36 months.", "IP indemnity is uncapped."],
                clause_ids=["liability-cap-1"],
            )
        ],
        open_questions=["Is a DPA incorporated anywhere?"],
    ),
    validation=__import__("summarizer.schemas", fromlist=["ValidationReport"]).ValidationReport(
        findings=[
            PlaybookFinding(
                rule_id="LIAB-01",
                rule_title="Liability cap",
                status="deviation",
                severity="high",
                clause_ids=["liability-cap-1"],
                explanation="3x fees over 36 months vs the 12-month cap required.",
                suggested_redline="Replace with a 12-month fees cap.",
            ),
            PlaybookFinding(
                rule_id="PAY-01",
                rule_title="Payment terms",
                status="deviation",
                severity="medium",
                clause_ids=["payment-1"],
                explanation="Net 90 exceeds the net 60 fallback.",
            ),
        ],
        blocking_count=1,
        overall="fail",
    ),
    judge=JudgeVerdict(score=0.62, grounded=False, coverage=0.7, issues=[], verdict="revise"),
    attempts=2,
    grounding_rate=rate,
    needs_human_review=True,
    usd_cost=0.0,
)

packet = render_review_packet(result)
assert "UNVERIFIED" in packet and "Human validation" in packet and "LIAB-01" in packet
print(f"review packet OK  {len(packet.splitlines())} lines")
print("\nall offline stages pass")
