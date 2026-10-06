"""Offline test of the backend. No API key, no cloud account, no network.

Covers the parts most likely to break: the grounding quote-to-box algorithm,
the state DAG's scoped invalidation, storage round-trips, rule-major validation
fan-out, and a full run through the API with a stubbed OCR provider and LLM.

    python backend_test.py
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import tempfile
import uuid

# Point every provider at something local BEFORE importing settings.
_TMP = tempfile.mkdtemp(prefix="contract-idp-test-")
os.environ.update(
    STORAGE_PROVIDER="local",
    STORAGE_LOCAL_ROOT=os.path.join(_TMP, "storage"),
    OCR_PROVIDER="none",
    PIPELINE_DOCTYPE_CONFIG_DIR="./doctypes",
    PIPELINE_PLAYBOOK_SEED_DIR="./playbook",
    DB_URL="sqlite+aiosqlite:///" + os.path.join(_TMP, "test.db").replace("\\", "/"),
    DB_AUTO_CREATE="true",
    QUOTA_ENABLED="false",
    LOG_LEVEL="WARNING",
)

import litellm  # noqa: E402
from litellm.types.utils import ChatCompletionMessageToolCall as ToolCall  # noqa: E402
from litellm.types.utils import Choices, Function, Message, ModelResponse  # noqa: E402
from pypdf import PdfWriter  # noqa: E402

from src.pipeline.stages import grounding as grounding_stage  # noqa: E402
from src.schemas.documents import (  # noqa: E402
    AnalysisResult,
    BoundingBox,
    Clause,
    OCRPage,
    OCRResult,
    OCRWord,
    Risk,
)
from src.schemas.state import RunState, StageStatus  # noqa: E402
from src.services.ocr.base import OCRProvider  # noqa: E402

PASS, FAIL = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASS if condition else FAIL).append(name)
    print(f"  {'OK  ' if condition else 'FAIL'} {name}" + (f"  {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# 1. grounding: quote -> char span -> word boxes, one box per line
# ---------------------------------------------------------------------------
print("\n[1] grounding")


def _page_with_words(text_lines, page=1):
    """Build an OCRPage whose word offsets are exact and boxes sit on lines."""
    words, parts, cursor = [], [], 0
    for line_index, line in enumerate(text_lines):
        for token in line.split():
            if parts:
                parts.append(" ")
                cursor += 1
            start = cursor
            parts.append(token)
            cursor += len(token)
            words.append(
                OCRWord(
                    text=token, start=start, end=cursor, confidence=0.95,
                    box=BoundingBox(
                        left=0.1, top=0.1 + line_index * 0.05,
                        right=0.9, bottom=0.14 + line_index * 0.05,
                    ),
                )
            )
        parts.append("\n")
        cursor += 1
    page_obj = OCRPage(page=page, text="".join(parts), words=words, width=612, height=792)
    page_obj.mean_confidence = 0.95
    return page_obj


ocr = OCRResult(
    doc_id="sha256:test",
    provider="azure",
    pages=[
        _page_with_words(["MASTER SERVICES AGREEMENT", "dated 14 March 2025"], 1),
        _page_with_words(
            [
                "6. LIMITATION OF LIABILITY",
                "Supplier's aggregate liability shall not exceed",
                "three times the fees paid in the preceding year.",
            ],
            2,
        ),
    ],
)

analysis = AnalysisResult(
    clauses=[
        Clause(
            id="liability-1", category="Limitation of Liability", heading="Liability cap",
            summary="Capped at 3x fees.", risk=Risk.high, rationale="Above the ceiling.",
            quote="aggregate liability shall not exceed", page=2,
        ),
        Clause(  # whitespace and case differ -> must still resolve
            id="liability-2", category="Limitation of Liability", heading="Cap multiple",
            summary="3x.", risk=Risk.medium, rationale="n/a",
            quote="three   times the FEES paid", page=2,
        ),
        Clause(  # mislabelled page -> must be found anyway and corrected
            id="title-1", category="Preamble", heading="Title",
            summary="MSA.", risk=Risk.low, rationale="n/a",
            quote="MASTER SERVICES AGREEMENT", page=2,
        ),
        Clause(  # not in the document at all
            id="ghost-1", category="Insurance", heading="Insurance",
            summary="Ghost.", risk=Risk.low, rationale="n/a",
            quote="Supplier shall maintain insurance of ten million dollars", page=2,
        ),
    ]
)

result = grounding_stage.run(analysis, ocr)
by_id = {g.clause_id: g for g in result.clauses}

check("exact quote resolves", by_id["liability-1"].match == "exact")
check("whitespace/case variant resolves", by_id["liability-2"].match in {"exact", "fuzzy"},
      by_id["liability-2"].match)
check("hallucinated quote is missing", by_id["ghost-1"].match == "missing")
check("mislabelled page corrected 2 -> 1",
      by_id["title-1"].regions and by_id["title-1"].regions[0].page == 1)
check("boxes produced for a located quote", len(by_id["liability-1"].regions[0].boxes) >= 1)
check("multi-line quote yields one box per line",
      len(by_id["liability-2"].regions[0].boxes) >= 1,
      f"{len(by_id['liability-2'].regions[0].boxes)} boxes")
check("grounding rate excludes the ghost", abs(result.rate - 0.75) < 1e-6, f"{result.rate:.2f}")
check("clause offsets written back", analysis.clauses[0].char_start is not None)
check("min confidence surfaced", by_id["liability-1"].min_confidence == 0.95)

# ---------------------------------------------------------------------------
# 2. state DAG: scoped invalidation
# ---------------------------------------------------------------------------
print("\n[2] state DAG / scoped retry")

from src.pipeline.orchestrator import (  # noqa: E402
    S_ANALYSIS, S_OCR, S_TYPE, S_VALIDATION, build_dag,
)

state = build_dag(RunState(run_id="r1", doc_id="d1", filename="x.pdf"))
for record in state.stages.values():
    record.status = StageStatus.done

invalidated = state.invalidate_from(S_TYPE)
check("retry from typedetect leaves OCR alone", state.get(S_OCR).status == StageStatus.done)
check("retry from typedetect invalidates analysis", S_ANALYSIS in invalidated)
check("retry from typedetect invalidates validation", S_VALIDATION in invalidated)
check("retry from typedetect does not invalidate preprocess", "preprocess" not in invalidated)

for record in state.stages.values():
    record.status = StageStatus.done
only_validation = state.invalidate_from(S_VALIDATION)
check("retry from validation is narrow",
      set(only_validation) == {S_VALIDATION, "consolidate"}, str(sorted(only_validation)))
check("ready() respects dependencies",
      {r.key for r in state.ready()} == {S_VALIDATION})

# ---------------------------------------------------------------------------
# 3. storage round-trip + key layout
# ---------------------------------------------------------------------------
print("\n[3] storage")

from src.services.storage.base import StorageService  # noqa: E402
from src.services.storage.factory import get_storage  # noqa: E402


async def _storage_checks():
    storage = get_storage()
    key = StorageService.source_key("sha256:abc")
    await storage.save_bytes(b"%PDF-1.4 hello", key, content_type="application/pdf")
    check("save + load round-trips", await storage.load_bytes(key) == b"%PDF-1.4 hello")
    check("exists is true after save", await storage.exists(key))
    await storage.save_json({"a": 1}, "runs/r1/state.json")
    check("json round-trips", (await storage.load_json("runs/r1/state.json"))["a"] == 1)
    check("list_keys finds what we wrote", "runs/r1/state.json" in await storage.list_keys("runs/"))
    check("delete reports miss on absent key", await storage.delete("nope/missing.json") is False)

    from src.core.errors.exceptions import StorageError
    try:
        await storage.load_bytes("../../etc/passwd")
        check("path traversal refused", False)
    except StorageError:
        check("path traversal refused", True)
    except Exception as exc:
        check("path traversal refused", False, type(exc).__name__)


asyncio.run(_storage_checks())

# ---------------------------------------------------------------------------
# 4. full run through the API, with OCR and the LLM stubbed
# ---------------------------------------------------------------------------
print("\n[4] end-to-end through the API")

writer = PdfWriter()
for _ in range(3):
    writer.add_blank_page(width=612, height=792)
buffer = io.BytesIO()
writer.write(buffer)
PDF_BYTES = buffer.getvalue()


class StubOCR(OCRProvider):
    """Returns the fixture pages above, so offsets and boxes are real."""

    name = "azure"
    calls = 0

    async def analyze_page(self, pdf_bytes, page_number):
        StubOCR.calls += 1
        return _page_with_words([f"Page {page_number} clause text here."], page_number)

    async def analyze_document(self, pdf_bytes):
        StubOCR.calls += 1
        return [
            _page_with_words(["MASTER SERVICES AGREEMENT", "dated 14 March 2025"], 1),
            _page_with_words(
                ["Supplier's aggregate liability shall not exceed", "three times the fees paid."], 2
            ),
            _page_with_words(["Invoices are payable within ninety (90) days."], 3),
        ]


TOOL_REPLIES = {
    "record_document_type": {
        "type": "msa", "confidence": 0.94, "alternatives": [],
        "evidence": "Titled Master Services Agreement.", "evidence_page": 1,
    },
    "record_clauses": {
        "clauses": [
            {
                "id": "liability-1", "category": "Limitation of Liability",
                "heading": "Liability cap", "summary": "Capped at three times fees.",
                "risk": "high", "rationale": "Above the 12-month ceiling.",
                "quote": "aggregate liability shall not exceed", "page": 2,
            },
            {
                "id": "payment-1", "category": "Payment Terms", "heading": "Net 90",
                "summary": "Invoices payable in 90 days.", "risk": "medium",
                "rationale": "Beyond net 60.",
                "quote": "Invoices are payable within ninety (90) days.", "page": 3,
            },
        ]
    },
    "record_summary": {
        "title": "Master Services Agreement", "parties": ["Northwind", "Calder"],
        "document_type": "msa", "effective_date": "14 March 2025", "term": "24 months",
        "governing_law": "Singapore",
        "executive_summary": "Supplier-unfavourable MSA with a 3x cap and net-90 terms.",
        "sections": [
            {"heading": "Liability & Indemnity", "bullets": ["Cap is 3x fees."],
             "clause_ids": ["liability-1"]},
            {"heading": "Invented Heading", "bullets": ["Net 90."], "clause_ids": ["payment-1"]},
        ],
        "open_questions": ["Is a DPA incorporated?"],
    },
    "record_finding": {
        "rule_id": "PLACEHOLDER", "rule_title": "x", "status": "deviation",
        "severity": "high", "clause_ids": ["liability-1"],
        "explanation": "Exceeds the ceiling.", "suggested_redline": "Cap at 12 months.",
    },
}
LLM_CALLS = {"count": 0, "tools": []}


async def fake_acompletion(**kwargs):
    name = kwargs["tools"][0]["function"]["name"]
    LLM_CALLS["count"] += 1
    LLM_CALLS["tools"].append(name)
    payload = dict(TOOL_REPLIES[name])
    call = ToolCall(
        id=f"toolu_{uuid.uuid4().hex[:8]}", type="function",
        function=Function(name=name, arguments=json.dumps(payload)),
    )
    return ModelResponse(
        id="m", model="gpt-4o",
        choices=[Choices(finish_reason="tool_calls", index=0,
                         message=Message(role="assistant", content=None, tool_calls=[call]))],
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    )


litellm.acompletion = fake_acompletion

# Patch the name where it is USED, not where it is defined - ocr_stage does
# `from ... import get_ocr_provider`, so it holds its own reference.
import src.pipeline.stages.ocr_stage as ocr_stage_mod  # noqa: E402
import src.services.ocr.factory as ocr_factory  # noqa: E402

_STUB = StubOCR()
ocr_factory.get_ocr_provider.cache_clear()
ocr_stage_mod.get_ocr_provider = lambda: _STUB

from fastapi.testclient import TestClient  # noqa: E402
from main import app  # noqa: E402

client = TestClient(app)  # lifespan creates the tables

# Seed the msa rules the way the seed script does, so validation has rows.
from scripts.seed_rules import parse_playbook  # noqa: E402
from src.core.database.connection import get_session_factory  # noqa: E402
from src.repositories.validation_rule_repository import ValidationRuleRepository  # noqa: E402


async def _seed():
    rows = parse_playbook(__import__("pathlib").Path("seeds/playbooks/default_playbook.yaml"), "msa")
    async with get_session_factory()() as session:
        return await ValidationRuleRepository(session).upsert_many(rows)


with TestClient(app):  # trigger lifespan -> create_all
    seeded = asyncio.run(_seed())
check("legacy playbook seeds into the rules table", seeded["created"] == 10, str(seeded))

response = client.get("/api/v1/health")
check("health responds", response.status_code == 200, response.json().get("status", ""))

response = client.get("/api/v1/config/doctypes")
check("doctypes load from config", response.json()["count"] >= 2,
      str([t["slug"] for t in response.json()["types"]]))

upload = client.post(
    "/api/v1/documents",
    files={"file": ("msa.pdf", PDF_BYTES, "application/pdf")},
)
check("upload accepted", upload.status_code == 201, str(upload.status_code))
doc_id = upload.json()["doc_id"]
check("doc_id is a content hash", doc_id.startswith("sha256:"))
check("quality scan flags the scanned pages", len(upload.json()["flagged_pages"]) == 3,
      str(upload.json()["flagged_pages"]))

upload2 = client.post("/api/v1/documents", files={"file": ("copy.pdf", PDF_BYTES, "application/pdf")})
check("same bytes give the same doc_id", upload2.json()["doc_id"] == doc_id)

created = client.post("/api/v1/runs", json={"doc_id": doc_id, "config": {}})
check("run accepted", created.status_code == 202, str(created.status_code))
run_id = created.json()["run_id"]

run = client.get(f"/api/v1/runs/{run_id}").json()
check("run completed", run["status"] == "completed", f"{run['status']} / {run.get('error')}")
check("every stage recorded", len(run["stages"]) == 8, str(len(run["stages"])))
check("cost attributed per stage", any(s["cost_usd"] >= 0 for s in run["stages"]))

result = client.get(f"/api/v1/runs/{run_id}/result").json()
check("result has clauses", len(result["analysis"]["clauses"]) == 2)
check("grounding ran", result["grounding"] is not None)
check("grounding located the quotes", result["grounding_rate"] > 0.0,
      f"{result['grounding_rate']:.2f}")
check("validation ran against the msa playbook",
      result["validation"] and len(result["validation"]["findings"]) > 0,
      str(len(result["validation"]["findings"]) if result["validation"] else 0))
check("one LLM call per rule (rule-major)",
      LLM_CALLS["tools"].count("record_finding") == len(result["validation"]["findings"]),
      f"{LLM_CALLS['tools'].count('record_finding')} calls")
check("off-taxonomy heading folded back into the configured set",
      all(s["heading"] in
          ["Parties & Term", "Commercials", "Liability & Indemnity", "IP & Data",
           "Termination", "Other"]
          for s in result["segmentation"]["sections"]),
      str([s["heading"] for s in result["segmentation"]["sections"]]))

review = client.get(f"/api/v1/runs/{run_id}/review")
check("review markdown renders", review.status_code == 200 and "# Review packet" in review.text)
check("review uses the msa checklist",
      "Liability cap and indemnity carve-outs confirmed" in review.text)

# --- scoped retry: OCR must not re-run ---------------------------------
ocr_calls_before = StubOCR.calls
retry = client.post(
    f"/api/v1/runs/{run_id}/retry",
    json={"stage": "typedetect", "reason": "wrong_type", "config": {"document_type": "msa"}},
)
check("retry accepted", retry.status_code == 202, str(retry.status_code))
check("retry invalidated the downstream stages",
      "analysis" in retry.json()["invalidated"] and "ocr" not in retry.json()["invalidated"],
      str(sorted(retry.json()["invalidated"])))
after = client.get(f"/api/v1/runs/{run_id}").json()
check("run completed again", after["status"] == "completed", after["status"])
check("OCR was NOT re-run on a scoped retry", StubOCR.calls == ocr_calls_before,
      f"{StubOCR.calls} vs {ocr_calls_before}")

check("unknown stage rejected",
      client.post(f"/api/v1/runs/{run_id}/retry", json={"stage": "nope"}).status_code == 422)
check("missing run is a 404", client.get("/api/v1/runs/does-not-exist").status_code == 404)
check("non-PDF upload rejected",
      client.post("/api/v1/documents",
                  files={"file": ("x.txt", b"hi", "text/plain")}).status_code == 422)


# --- rules CRUD -------------------------------------------------------
print("\n[5] validation rules in the database")

listed = client.get("/api/v1/rules", params={"doctype": "msa"}).json()
check("rules readable by doctype", listed["count"] == 10, str(listed["count"]))

created_rule = client.post("/api/v1/rules", json={
    "code": "TEST-01", "name": "Test rule", "rule": "Something must hold.",
    "doctype": "msa", "severity": "low",
})
check("rule created", created_rule.status_code == 201, str(created_rule.status_code))
rule_id = created_rule.json()["id"]

dupe = client.post("/api/v1/rules", json={
    "code": "TEST-01", "name": "Dup", "rule": "x", "doctype": "msa",
})
check("duplicate (doctype, code) rejected", dupe.status_code == 422, str(dupe.status_code))

patched = client.patch(f"/api/v1/rules/{rule_id}", json={"enabled": False})
check("rule disabled via PATCH", patched.json()["enabled"] is False)

enabled_only = client.get("/api/v1/rules", params={"doctype": "msa", "enabled_only": True}).json()
check("disabled rule excluded from the enabled set", enabled_only["count"] == 10,
      str(enabled_only["count"]))

globals_ = client.post("/api/v1/rules", json={
    "code": "GLOBAL-01", "name": "Applies everywhere", "rule": "Must be signed.",
    "severity": "high",
})
check("global rule created (doctype null)", globals_.status_code == 201)
both = client.get("/api/v1/rules", params={"doctype": "employment_offer"}).json()
check("global rule reaches another doctype",
      any(r["code"] == "GLOBAL-01" for r in both["rules"]), str(both["count"]))
check("msa rules do NOT leak into employment_offer",
      not any(r["code"] == "LIAB-01" for r in both["rules"]))

bulk = client.post("/api/v1/rules/bulk", json={"rules": [
    {"code": "TEST-01", "name": "Renamed", "rule": "Updated text.", "doctype": "msa"},
    {"code": "TEST-02", "name": "New one", "rule": "Another.", "doctype": "msa"},
]}).json()
check("bulk upsert updates and inserts", bulk["created"] == 1 and bulk["updated"] == 1, str(bulk))

check("delete removes the rule",
      client.delete(f"/api/v1/rules/{rule_id}").status_code == 204)
check("deleted rule is a 404",
      client.get(f"/api/v1/rules/{rule_id}").status_code == 404)
check("no YAML playbook loader remains",
      not hasattr(__import__("src.services.doctypes", fromlist=["x"]), "load_playbook"))

# ---------------------------------------------------------------------------
shutil.rmtree(_TMP, ignore_errors=True)
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
    raise SystemExit(1)
print("backend OK")
