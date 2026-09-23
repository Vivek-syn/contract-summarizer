"""Playbook / rulebook loading.

A playbook is a YAML list of rules. Each rule becomes both a retrievable chunk
(for the validator's `search_playbook` tool) and an item on the checklist the
validator must cover.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from .retrieval import Index, index_of_texts

REQUIRED = ("id", "title", "rule")


def load(path: str | Path) -> Tuple[List[Dict[str, str]], Index]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    rules = data.get("rules", [])
    if not rules:
        raise ValueError(f"{path}: no `rules:` found")

    cleaned: List[Dict[str, str]] = []
    for i, rule in enumerate(rules):
        missing = [k for k in REQUIRED if not rule.get(k)]
        if missing:
            raise ValueError(f"{path}: rule #{i} is missing {missing}")
        cleaned.append({k: str(v) for k, v in rule.items()})

    entries = [
        (
            rule["id"],
            "\n".join(
                filter(
                    None,
                    [
                        f"RULE {rule['id']}: {rule['title']}",
                        f"Requirement: {rule['rule']}",
                        f"Fallback position: {rule['fallback']}" if rule.get("fallback") else "",
                        f"Severity if breached: {rule.get('severity', 'medium')}",
                        f"Preferred language: {rule['preferred_language']}" if rule.get("preferred_language") else "",
                    ],
                )
            ),
        )
        for rule in cleaned
    ]
    return cleaned, index_of_texts(entries)
