"""End-to-end run with a stubbed transport - no API key, no network.

Verifies the orchestration itself: the agent loop's tool dispatch and message
plumbing, the structured-output repair loop, the judge retry gate, and the
review packet. Run it before you spend money on a real document.

    python mock_test.py
"""
from __future__ import annotations

import json
import uuid

import litellm
from litellm.types.utils import Choices, Function, Message, ModelResponse
from litellm.types.utils import ChatCompletionMessageToolCall as ToolCall

from summarizer import pipeline
from summarizer.config import SETTINGS

# Pin the settings this test asserts on. The scripted turns below assume the
# judge loop runs twice, so a .env that sets SUMMARIZER_MAX_REPAIRS=0 (or a
# different threshold) must not decide whether the test passes.
SETTINGS.max_repairs = 2
SETTINGS.judge_threshold = 0.75

CALLS: list[dict] = []


def _reply(tool_name: str | None, args: dict | None = None, text: str | None = None) -> ModelResponse:
    tool_calls = None
    if tool_name:
        tool_calls = [
            ToolCall(
                id=f"toolu_{uuid.uuid4().hex[:12]}",
                type="function",
                function=Function(name=tool_name, arguments=json.dumps(args or {})),
            )
        ]
    message = Message(role="assistant", content=text, tool_calls=tool_calls)
    return ModelResponse(
        id="msg_mock",
        choices=[Choices(finish_reason="tool_calls" if tool_calls else "stop", index=0, message=message)],
        model="claude-opus-5",
        usage={"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200},
    )


CLAUSES = {
    "clauses": [
        {
            "id": "liability-cap-1",
            "category": "Limitation of Liability",
            "heading": "Aggregate liability cap",
            "summary": "Capped at 3x fees paid over the preceding 36 months.",
            "risk": "high",
            "rationale": "Three times the playbook's 12-month ceiling.",
            "spans": [{"quote": "shall not exceed three (3) times the total fees paid", "page": 1}],
        },
        {
            "id": "payment-1",
            "category": "Payment Terms",
            "heading": "Net 90",
            "summary": "Invoices payable within 90 days.",
            "risk": "medium",
            "rationale": "Beyond the net 60 fallback.",
            "spans": [{"quote": "Invoices are payable within ninety (90) days of receipt", "page": 1}],
        },
    ]
}

SUMMARY = {
    "title": "Northwind / Calder Master Services Agreement",
    "parties": ["Northwind Analytics Ltd", "Calder Retail Group plc"],
    "agreement_type": "Master Services Agreement",
    "effective_date": "14 March 2025",
    "term": "24 months, auto-renewing for 24-month periods",
    "governing_law": "Republic of Singapore",
    "executive_summary": "A supplier-unfavourable MSA: liability capped at 3x fees over 36 months, net-90 payment, perpetual confidentiality and Singapore law.",
    "sections": [
        {
            "heading": "Liability & Indemnity",
            "bullets": ["Cap is 3x fees over 36 months.", "The IP indemnity sits outside the cap."],
            "clause_ids": ["liability-cap-1"],
        },
        {"heading": "Commercials", "bullets": ["Net 90 payment terms."], "clause_ids": ["payment-1"]},
    ],
    "open_questions": ["No DPA is referenced - is one incorporated elsewhere?"],
}

FINDINGS = [
    {
        "rule_id": "LIAB-01",
        "rule_title": "Liability cap",
        "status": "deviation",
        "severity": "high",
        "clause_ids": ["liability-cap-1"],
        "explanation": "3x fees over 36 months against a 12-month ceiling.",
        "suggested_redline": "Cap at total fees paid in the preceding 12 months.",
    },
    {
        "rule_id": "PAY-01",
        "rule_title": "Payment terms",
        "status": "deviation",
        "severity": "medium",
        "clause_ids": ["payment-1"],
        "explanation": "Net 90 exceeds the net 60 fallback.",
    },
    {
        "rule_id": "DATA-01",
        "rule_title": "Data protection",
        "status": "missing",
        "severity": "high",
        "clause_ids": [],
        "explanation": "No DPA and no 72-hour breach notification.",
    },
]

# Scripted turns, in the order the pipeline asks for them. The validator turn
# deliberately mixes retrieval calls with findings to exercise the loop.
SCRIPT = [
    ("record_clauses", CLAUSES),
    ("record_summary", SUMMARY),
    ("search_playbook", {"query": "liability cap"}),          # validator turn 1
    ("search_document", {"query": "limitation of liability"}),  # validator turn 2
    ("record_finding", FINDINGS[0]),
    ("record_finding", FINDINGS[1]),
    ("record_finding", FINDINGS[2]),
    (None, None),  # validator says DONE
    ("record_verdict", {"score": 0.55, "grounded": True, "coverage": 0.6,
                        "issues": [{"stage": "clauses", "severity": "high",
                                    "detail": "Confidentiality and governing law were never extracted.",
                                    "fix_instruction": "Extract the confidentiality and governing law clauses."}],
                        "verdict": "revise"}),
    # --- attempt 2, after judge feedback ---
    ("record_clauses", CLAUSES),
    ("record_summary", SUMMARY),
    ("record_finding", FINDINGS[0]),
    ("record_finding", FINDINGS[1]),
    ("record_finding", FINDINGS[2]),
    (None, None),
    ("record_verdict", {"score": 0.88, "grounded": True, "coverage": 0.9,
                        "issues": [], "verdict": "accept"}),
]


def fake_completion(**kwargs):
    step = len(CALLS)
    CALLS.append({"tools": [t["function"]["name"] for t in kwargs.get("tools", [])]})
    if step >= len(SCRIPT):
        raise AssertionError(f"pipeline made {step + 1} calls; script has {len(SCRIPT)}")
    name, args = SCRIPT[step]
    return _reply(name, args, text=None if name else "DONE")


litellm.completion = fake_completion

result = pipeline.run(
    "samples/sample_msa.txt",
    "playbook/default_playbook.yaml",
    on_progress=lambda m: print(m, flush=True),
)

print("\n--- assertions ---")
assert result.attempts == 2, result.attempts
assert result.judge.verdict == "accept" and result.judge.score == 0.88
assert len(result.validation.findings) == 3, result.validation.findings
assert result.validation.overall == "fail" and result.validation.blocking_count == 2
assert result.grounding_rate == 1.0, result.grounding_rate
assert len(CALLS) == len(SCRIPT), (len(CALLS), len(SCRIPT))

packet = pipeline.render_review_packet(result)
assert "DATA-01" in packet and "Human validation" in packet
print(f"calls={len(CALLS)} attempts={result.attempts} findings={len(result.validation.findings)} "
      f"grounding={result.grounding_rate:.0%} packet={len(packet.splitlines())} lines")
print("mock end-to-end OK")
