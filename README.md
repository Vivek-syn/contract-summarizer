# Contract IDP

FastAPI backend for document understanding: OCR, clause analysis, grounding,
type-aware segmentation and playbook validation.

Architecture doc: [Contract IDP Pipeline Architecture](https://claude.ai/artifact/MxdwzmB8EybR4sDL9yiH6j)

## Run it

```bash
pip install -r requirements.txt          # add -azure or -aws for those providers
cp .env.example .env                     # set OCR_AZURE_* and ANTHROPIC_API_KEY
alembic upgrade head                     # create the validation_rules table
python -m scripts.seed_rules             # import the legacy YAML playbooks, once
uvicorn main:app --reload
```

SQLite by default, so no database server is needed. Point `DB_URL` at
`postgresql+asyncpg://...` for a deployment.

`http://localhost:8000/docs` for the OpenAPI UI, `/api/v1/health` for provider status.

Offline tests — no API key, no cloud account, no network:

```bash
python backend_test.py
```

## The pipeline

```
PDF ─► preprocess ─► OCR ─┬─► type detection ─► analysis ─┬─► grounding
       hash, split,       │   (pages 1-2,                 ├─► segmentation
       quality scan       │    own call)                  └─► validation
                          │                                   (rule-major)
                          └── cached by content hash              │
                                                                  ▼
                                                            consolidate
                                                         result.json + review.md
```

| Stage | Model? | Notes |
|---|---|---|
| preprocess | no | hash, page split, quality flags. Flags never block |
| ocr | no | Azure DI or Textract; cached by document content hash |
| type detection | yes (fast) | reads pages 1-2 only; a closed enum + `unknown` |
| analysis | yes | reads OCR **text**, never the PDF |
| grounding | no | quote → char span → word boxes, one box per line |
| segmentation | yes | section taxonomy comes from the doc-type config |
| validation | yes | one call per rule, run in parallel |
| consolidate | no | assembles JSON, renders the review packet |

## Why it is shaped this way

**The model never sees the PDF.** OCR writes the text, the model writes the
quotes, and grounding checks one against the other. When the model produced both,
a hallucination landed in both halves consistently and the verifier confirmed it —
a closed loop marking its own homework.

**Type detection is its own stage.** It is a routing decision: wrong here and the
segmentation schema and the playbook are both wrong, with nothing downstream able
to tell. Isolated, it can be retried alone, cached, and overridden in one click.

**Validation is rule-major, not page-wise.** A rule needs its relevant clauses
wherever they live. Validating page 7 in isolation asks a question with no answer
and forces a consolidation step to reconcile N partial verdicts per rule.

**Grounding is deterministic.** The only stage that cannot hallucinate, which is
why everything else is anchored to it. A quote it cannot locate is reported as
unverified rather than quietly dropped.

## Scoped retry

The human gate is not approve/reject — it is *which node do I invalidate*.

```bash
curl -X POST localhost:8000/api/v1/runs/$RUN/retry \
  -d '{"stage":"typedetect","reason":"wrong_type","config":{"document_type":"msa"}}'
```

| Reviewer sees | Stage to name | Cost |
|---|---|---|
| Wrong document type | `typedetect` | cents |
| Missed a clause | `analysis` | cents |
| Bad quote or box | `grounding` | negligible |
| Wrong finding | `validation` | negligible |
| Bad OCR | `ocr` | full re-OCR |

OCR only re-runs when named, and even then a content-hash cache hit makes it
free. `backend_test.py` asserts this.

## Configuration

Providers are flags, resolved through factories. Adding a backend is one module
plus one registry entry — call sites never branch.

| Setting | Values | Default |
|---|---|---|
| `OCR_PROVIDER` | `azure` \| `aws` \| `none` | `azure` |
| `STORAGE_PROVIDER` | `local` \| `azure` \| `s3` | `local` |
| `LLM_MODEL` | any LiteLLM id | `anthropic/claude-opus-5` |

Full list in `.env.example`.

### Document types

One YAML per type in `doctypes/`, driving three things that used to be hardcoded
in three places:

```yaml
slug: employment_offer
focus: [...]              # what analysis emphasises
sections: [...]           # the segmentation taxonomy, enforced after the call
review_checklist: [...]   # the sign-off list in review.md
```

Adding a document type is a file, not a code change. The type detector builds its
catalogue from the same files.

### Validation rules

Rules are **rows in `validation_rules`**, not files. They are edited through the
API by the people who own the policy, without a release:

```bash
curl localhost:8000/api/v1/rules?doctype=msa
curl -X POST localhost:8000/api/v1/rules -d '{
  "code":"LIAB-01","name":"Liability cap","doctype":"msa","severity":"high",
  "rule":"Aggregate liability must be capped at 12 months of fees."}'
curl -X PATCH localhost:8000/api/v1/rules/$ID -d '{"enabled":false}'
```

Scoping is by `doctype`. A rule with a doctype applies to that type only; a rule
with `doctype = NULL` applies to every document. That is what replaced "which
playbook file do I load" — and it is why an MSA rule can no longer silently run
against an offer letter.

`seeds/playbooks/*.yaml` are the original playbooks, kept as an import source
only. Nothing reads them at runtime; `scripts/seed_rules.py` upserts them on
`(doctype, code)`, so re-running updates rather than duplicating.

## Layout

```
main.py                     FastAPI app, error handling, lifespan
src/
  api/v1/endpoints/         documents.py  runs.py  health.py
  api/v1/models/            requests.py  responses.py
  core/config/              settings.py, one BaseSettings per concern
  core/errors/              exceptions with HTTP status + code
  services/ocr/             base.py  azure_ocr.py  aws_ocr.py  factory.py
  services/storage/         base.py  local  azure  s3  factory.py
  services/llm/             adapter.py - LiteLLM + usage accounting
  services/state/           store.py - run persistence
  services/doctypes.py      per-type config
  models/database/          SQLAlchemy models
  repositories/             data access; nothing else touches a session
  core/database/            engine, session, Postgres/SQLite type shims
alembic/                    migrations
scripts/seed_rules.py       one-way YAML -> database import
seeds/playbooks/            the original YAML, import source only
  pipeline/stages/          one module per stage, pure JSON in/out
  pipeline/orchestrator.py  DAG resolution, persistence, parallel branches
  schemas/                  stage contracts + run state
  utils/retrieval.py        BM25, dependency-free
```

## Provider notes

Three LiteLLM behaviours are handled in `services/llm/adapter.py`, each verified
against LiteLLM's own transforms rather than assumed:

- `output_config` must be a **top-level kwarg** for Anthropic. As `extra_body` it
  is sent as a literal `extra_body` key, which Anthropic rejects; for OpenAI it is
  not dropped but tucked into `extra_body`, which OpenAI rejects.
- `reasoning_effort` sent to **Anthropic** is rewritten to the deprecated
  `thinking: {type: "enabled", budget_tokens: N}` — a 400 on current models.
- `strict` is stripped for OpenAI, whose strict mode requires every property in
  `required`; our schemas have genuinely optional fields.

Structured output forces the tool call on OpenAI and uses `auto` plus an explicit
instruction on Anthropic, whose newest models reject forced tool choice.

## Not done yet

- **Workers and Redis.** Runs execute as FastAPI background tasks. Stages are
  already pure functions of JSON, so moving them behind a queue is a transport
  change, not a rewrite.
- **Per-type segmentation schemas** beyond the two in `doctypes/`.
- **No live call has been made through this backend.** Every test stubs the OCR
  provider and the LLM transport. The Azure and Textract response parsers are
  written against their documented shapes but have not seen a real response.
- **No OCR fallback.** If the configured provider fails, the run fails at the
  `ocr` stage and says so. The old model-transcription path was removed with the
  legacy CLI; re-adding it means implementing a provider, not flipping a flag.
- **Reviewer corrections are recorded, not used.** `retry.reason` is stored on the
  stage record. Nothing reads it yet — but a labelled failure case cannot be
  reconstructed after the fact.
