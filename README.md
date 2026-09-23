# Agentic contract summarizer — LiteLLM → Anthropic

An implementation of the whiteboard pipeline: parse a contract, extract clauses with
verbatim grounding, produce a structured summary, validate it against a playbook with
a tool-driven agent, gate it behind an LLM judge, and hand a packet to a human.

```
 User Doc
    │
 PDF parser / OCR ──────────── whole-doc fallback (PDF → Claude directly)
    │  text (extracted, preferred)
 chunk + BM25 index  ← the RAG box
    │
    ├──► extract_clauses ──► grounding verifier (deterministic, no LLM)
    │         Grounding arm         │ every quote located in the source, or flagged
    │                               ▼
    ├──► segment_summary ◄──────────┘
    │         Segmentation arm → StructuredSummary
    │
 validate (agentic: search_document / search_playbook / get_clause / record_finding)
    │         → ValidationReport against playbook + rulebook
    │
 judge ──── revise? ──► re-run every stage with the judge's fix instructions (bounded)
    │
 accept
    │
 human validation packet (Markdown, with unverified quotes flagged)
```

## Run it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python run.py samples/sample_msa.txt --playbook playbook/default_playbook.yaml -v
```

Outputs `out/<name>.review.md` (the human packet) and `out/<name>.result.json`
(the full typed result). Exit code is `1` when the playbook fails or the judge
still wants revisions — so it drops into CI as a gate.

Two tests run with no API key and no network:

```bash
python smoke_test.py   # ingest, retrieval, playbook, grounding, packet rendering
python mock_test.py    # full orchestration against a stubbed transport
```

## Layout

| File | Role |
|---|---|
| `summarizer/ingest.py` | PDF text layer → OCR → whole-doc fallback; page-addressable text |
| `summarizer/retrieval.py` | Dependency-free BM25 index over document chunks and playbook rules |
| `summarizer/llm.py` | LiteLLM layer: `structured()`, `agent_loop()`, cost meter, param degradation |
| `summarizer/agents.py` | The four model stages and their system prompts |
| `summarizer/grounding.py` | Deterministic quote verification (exact → fuzzy → missing) |
| `summarizer/playbook.py` | YAML rulebook → checklist + retrievable chunks |
| `summarizer/pipeline.py` | Orchestration, judge retry loop, review-packet rendering |
| `summarizer/schemas.py` | Pydantic contracts; also the JSON Schemas sent as tools |

## Design notes

**Grounding is not an LLM call.** `grounding.py` locates every quoted span in the
source text by exact match, then fuzzy match. A span it cannot find is marked
`missing`, surfaced to the judge, and printed with `!!` in the review packet. This is
the only part of the pipeline that cannot hallucinate, so it anchors everything else.

**Structured output without forced tool use.** Anthropic rejects forced `tool_choice`
alongside extended thinking, and the newest models reject it outright. So `structured()`
uses `tool_choice="auto"` plus an explicit instruction naming the tool, then feeds any
pydantic `ValidationError` back to the model as a repair turn. LiteLLM's Anthropic
transform also drops the `strict` flag, so this repair loop — not the provider — is what
guarantees a schema-valid object.

**`output_config` is a top-level kwarg, not `extra_body`.** LiteLLM forwards unknown
kwargs into the Anthropic request body, but forwards `extra_body` as a literal
`extra_body` key, which Anthropic rejects. `complete()` also degrades gracefully: on a
400 it strips one optional param at a time (`thinking`, `output_config`, `tool_choice`)
and retries, so the code still runs against a proxy or an older model.

**Adaptive thinking only.** `budget_tokens` is rejected on Opus 5; `config.py` never
sets it. Depth is controlled by `SUMMARIZER_EFFORT` (`low`…`max`).

**Provider switching needs no code change.** Point `SUMMARIZER_MODEL` at
`openai/...` and set `OPENAI_API_KEY`; `run.py` derives the required env var from the
model string via `litellm.validate_environment`. `llm.py` adapts the request per
provider, because the cross-provider failures are quiet rather than loud:

| | Anthropic | OpenAI |
|---|---|---|
| Reasoning depth | `thinking: adaptive` + `output_config.effort` | `reasoning_effort` (`xhigh`/`max` → `high`) |
| Output cap | `max_tokens` | `max_completion_tokens` (LiteLLM maps it) |
| Tool `strict` | ignored by the provider | **stripped** — see below |

Two traps worth knowing if you extend this: `output_config` is *not* dropped for
OpenAI (LiteLLM puts it in `extra_body`, which OpenAI rejects), and `reasoning_effort`
sent to Anthropic is rewritten to the deprecated `thinking: {type: "enabled",
budget_tokens: N}`, which Opus 5 rejects. `strict` is stripped for OpenAI because its
strict mode requires every property in `required`, and `Span.char_start` /
`PlaybookFinding.suggested_redline` are genuinely optional.

**The judge is a gate, not a rubber stamp.** Below `SUMMARIZER_JUDGE_THRESHOLD` the
whole analysis re-runs with the judge's `fix_instruction`s injected into every stage's
prompt, up to `SUMMARIZER_MAX_REPAIRS` times. `needs_human_review` is hardcoded `True` —
the judge decides whether the packet is worth a human's time, never whether to skip them.

## Configuration

All optional; defaults shown. See `.env.example`.

| Variable | Default | Notes |
|---|---|---|
| `SUMMARIZER_MODEL` | `anthropic/claude-opus-5` | `<provider>/<model>`; no date suffix. `openai/...` also works |
| `SUMMARIZER_JUDGE_MODEL` | `anthropic/claude-opus-5` | Separate — can be a different provider entirely |
| `SUMMARIZER_EFFORT` | `high` | `low` \| `medium` \| `high` \| `xhigh` \| `max`; mapped per provider |
| `SUMMARIZER_THINKING` | `1` | Anthropic adaptive thinking; ignored elsewhere |
| `SUMMARIZER_MAX_TOKENS` | `16000` | Non-streaming default; raise with streaming |
| `SUMMARIZER_JUDGE_THRESHOLD` | `0.75` | Below this the pipeline retries |
| `SUMMARIZER_MAX_REPAIRS` | `2` | Judge retries, and `structured()` repair turns |

## Extending it

- **Semantic retrieval** — replace `Index` with a vector store; only `search()` is used.
- **Clause-type coverage check** — add a deterministic pass asserting each playbook
  category maps to at least one clause, and feed misses to the judge as an issue.
- **Streaming** — for `max_tokens` above ~32k, switch `complete()` to `litellm.completion(..., stream=True)`
  and accumulate; the Anthropic SDK requires streaming near the 128k output ceiling.
- **Caching** — the system prompts and the document block are stable across the judge
  retry loop, so a `cache_control` breakpoint after the document block pays for itself
  from the second attempt onward.

## Not covered

- Parallel tool calls are handled in `agent_loop()` but the mocked test only exercises
  one call per turn; verify against the real API before relying on it.
- The OCR path needs `poppler` and `tesseract` on `PATH` and is untested here — no
  scanned sample was available.
- `whole_doc` mode has no character offsets, so grounding degrades to fuzzy matching
  against whatever text the model echoes back.
