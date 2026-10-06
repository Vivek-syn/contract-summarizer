# Legacy playbooks - import source only

These YAML files are **not read at runtime**. Validation rules live in the
`validation_rules` table; these are kept solely so the original rule text can be
re-imported or diffed against what is in the database.

Import them once:

```bash
python -m scripts.seed_rules --dir seeds/playbooks
```

The import is idempotent - rules upsert on `(doctype, code)`, so re-running
updates rather than duplicating. After that, edit rules through
`/api/v1/rules`, not here: a change made in these files has no effect.

| File | Imported as doctype |
|---|---|
| `default_playbook.yaml` | `msa` |
| `offer_letter_playbook.yaml` | `employment_offer` |

The mapping lives in `DEFAULT_MAPPING` in `scripts/seed_rules.py`. A file that is
not listed needs `--doctype` passed explicitly - guessing which documents a
playbook governs is what let an MSA playbook run against an offer letter.
