"""Document-type configuration.

One YAML per document type, driving three things that were previously hardcoded
in three different places:

- which clauses analysis should emphasise (``focus``)
- the section taxonomy segmentation must use (``sections``)
- the review packet's sign-off checklist (``review_checklist``)

Adding a document type is a file, not a code change, and the type detector's
catalogue is built from the same files so a new type is immediately selectable.

Validation rules are NOT here - they live in the database (see
``src/models/database/validation_rule.py``) and are scoped by the same
``doctype`` slug these files declare.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from src.core.errors.exceptions import ConfigurationError
from src.core.logging.structured import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def load_doctypes(directory: str) -> Dict[str, dict]:
    """Load every document-type config in ``directory``, keyed by slug."""
    root = Path(directory)
    if not root.exists():
        logger.warning("doctype dir %s does not exist; type detection will have no catalogue", root)
        return {}

    types: Dict[str, dict] = {}
    for path in sorted(root.glob("*.y*ml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"{path}: invalid YAML: {exc}") from exc

        slug = data.get("slug") or path.stem
        data["slug"] = slug
        data.setdefault("name", slug.replace("_", " ").title())
        types[slug] = data

    logger.info("loaded %d document types from %s", len(types), root)
    return types


def catalogue(directory: str) -> List[dict]:
    """The type list handed to the detector."""
    return [
        {"slug": t["slug"], "name": t.get("name", t["slug"]), "hints": t.get("hints", "")}
        for t in load_doctypes(directory).values()
    ]


def get_doctype(directory: str, slug: str) -> Optional[dict]:
    """One type's config, or None when the slug is unknown (including 'unknown')."""
    return load_doctypes(directory).get(slug)
