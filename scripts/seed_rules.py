"""Import legacy YAML playbooks into the validation_rules table.

A one-way migration: YAML was the old storage, the database is the new one.
Run it once per playbook, then edit rules through the API.

    python -m scripts.seed_rules                          # every playbook, mapped by doctype
    python -m scripts.seed_rules --file seeds/playbooks/x.yaml --doctype msa
    python -m scripts.seed_rules --dry-run

Idempotent: rules are upserted on ``(doctype, code)``, so re-running updates
rather than duplicating.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# The doctype each legacy playbook file belongs to. A file not listed here needs
# --doctype, because guessing which documents a playbook governs is exactly the
# mistake that made an MSA playbook run against an offer letter.
DEFAULT_MAPPING: Dict[str, str] = {
    "default_playbook.yaml": "msa",
    "offer_letter_playbook.yaml": "employment_offer",
}


def parse_playbook(path: Path, doctype: Optional[str]) -> List[dict]:
    """Turn one YAML playbook into rule rows."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_rules = data.get("rules") or []
    if not raw_rules:
        raise SystemExit(f"{path}: no `rules:` found")

    rows: List[dict] = []
    for position, rule in enumerate(raw_rules):
        missing = [f for f in ("id", "title", "rule") if not rule.get(f)]
        if missing:
            raise SystemExit(f"{path}: rule #{position} is missing {missing}")

        severity = str(rule.get("severity", "medium")).lower()
        if severity not in {"low", "medium", "high"}:
            print(f"  ! {rule['id']}: severity {severity!r} is not valid, using 'medium'")
            severity = "medium"

        rows.append(
            {
                "code": str(rule["id"]).strip(),
                "name": str(rule["title"]).strip(),
                "rule": str(rule["rule"]).strip(),
                "doctype": doctype,
                # The id prefix is a natural grouping: LIAB-01 -> liab.
                "group_name": str(rule["id"]).split("-")[0].lower() or None,
                "fallback": str(rule["fallback"]).strip() if rule.get("fallback") else None,
                "preferred_language": (
                    str(rule["preferred_language"]).strip()
                    if rule.get("preferred_language")
                    else None
                ),
                "severity": severity,
                "enabled": True,
                "sort_order": position,
                "extra": {"imported_from": path.name, "playbook": data.get("name")},
            }
        )
    return rows


async def seed(files: List[tuple], dry_run: bool) -> int:
    from src.core.database.connection import create_all, dispose_engine, get_session_factory
    from src.repositories.validation_rule_repository import ValidationRuleRepository

    total = 0
    all_rows: List[dict] = []
    for path, doctype in files:
        rows = parse_playbook(path, doctype)
        print(f"{path.name} -> doctype={doctype or 'GLOBAL'}: {len(rows)} rules")
        for row in rows:
            print(f"    {row['code']:12} {row['severity']:6} {row['name']}")
        all_rows += rows
        total += len(rows)

    if dry_run:
        print(f"\n[dry run] {total} rules parsed, nothing written")
        return 0

    await create_all()
    async with get_session_factory()() as session:
        counts = await ValidationRuleRepository(session).upsert_many(all_rows)
    await dispose_engine()
    print(f"\n{counts['created']} created, {counts['updated']} updated ({total} total)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", help="A single playbook YAML to import")
    parser.add_argument(
        "--doctype",
        help="Document type these rules apply to. Omit for a global rule set.",
    )
    parser.add_argument("--dir", default="seeds/playbooks", help="Directory of playbooks")
    parser.add_argument("--dry-run", action="store_true", help="Parse and print, write nothing")
    args = parser.parse_args()

    if args.file:
        path = Path(args.file)
        if not path.exists():
            raise SystemExit(f"no such file: {path}")
        if args.doctype is None and path.name not in DEFAULT_MAPPING:
            raise SystemExit(
                f"{path.name} is not in the default mapping; pass --doctype explicitly "
                f"(or --doctype '' for a global rule set)"
            )
        doctype = args.doctype if args.doctype is not None else DEFAULT_MAPPING[path.name]
        files = [(path, doctype or None)]
    else:
        directory = Path(args.dir)
        if not directory.exists():
            raise SystemExit(f"no such directory: {directory}")
        files = []
        for path in sorted(directory.glob("*.y*ml")):
            doctype = args.doctype if args.doctype is not None else DEFAULT_MAPPING.get(path.name)
            if doctype is None and path.name not in DEFAULT_MAPPING:
                print(f"  skipping {path.name}: no doctype mapping (pass --doctype)")
                continue
            files.append((path, doctype or None))
        if not files:
            raise SystemExit(f"no playbooks to import from {directory}")

    return asyncio.run(seed(files, args.dry_run))


if __name__ == "__main__":
    sys.exit(main())
