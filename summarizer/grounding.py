"""The Grounding arm: verify every quoted span actually exists in the source.

This is deterministic, not another LLM call. A span the verifier cannot locate
is marked ``missing`` - the judge sees that and the human reviewer sees it in
red, which is the whole point of the grounding branch.
"""
from __future__ import annotations

import difflib
import re
from typing import List, Tuple

from .schemas import Clause, ClauseSet, Span

_WS = re.compile(r"\s+")


def _canon(text: str) -> str:
    return _WS.sub(" ", text.lower().replace("’", "'").replace("“", '"').replace("”", '"')).strip()


def _fuzzy_locate(needle: str, haystack: str, *, threshold: float = 0.82) -> Tuple[int, int, float] | None:
    """Best approximate location of ``needle`` in ``haystack``.

    OCR output and text-layer extraction both introduce noise, so an exact
    match failing does not mean the model hallucinated.
    """
    if not needle:
        return None
    matcher = difflib.SequenceMatcher(None, haystack, needle, autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size > 0]
    if not blocks:
        return None
    anchor = max(blocks, key=lambda b: b.size)
    start = max(anchor.a - anchor.b, 0)
    end = min(start + len(needle), len(haystack))
    ratio = difflib.SequenceMatcher(None, haystack[start:end], needle, autojunk=False).ratio()
    return (start, end, ratio) if ratio >= threshold else None


def verify_span(span: Span, canon_text: str, raw_text: str, page_at) -> Span:
    needle = _canon(span.quote)
    if not needle:
        span.match, span.verified = "missing", False
        return span

    exact = canon_text.find(needle)
    if exact != -1:
        span.char_start, span.char_end = exact, exact + len(needle)
        span.match, span.verified = "exact", True
    else:
        hit = _fuzzy_locate(needle, canon_text)
        if hit:
            span.char_start, span.char_end, _ = hit[0], hit[1], hit[2]
            span.match, span.verified = "fuzzy", True
        else:
            span.match, span.verified = "missing", False

    if span.verified and span.char_start is not None:
        # Trust the located offset over the model's page guess.
        span.page = page_at(span.char_start)
    return span


def verify_clauses(clause_set: ClauseSet, doc) -> Tuple[ClauseSet, float, List[Clause]]:
    """Annotate every span and return ``(clauses, grounding_rate, ungrounded)``."""
    raw = doc.text
    canon = _canon(raw)
    # Map canonical offsets back to page numbers approximately via ratio.
    scale = (len(raw) / len(canon)) if canon else 1.0

    def page_at(canon_index: int) -> int:
        return doc.page_at(int(canon_index * scale))

    total = verified = 0
    ungrounded: List[Clause] = []
    for clause in clause_set.clauses:
        for span in clause.spans:
            verify_span(span, canon, raw, page_at)
            total += 1
            verified += int(span.verified)
        if clause.spans and not any(s.verified for s in clause.spans):
            ungrounded.append(clause)

    rate = (verified / total) if total else 0.0
    return clause_set, rate, ungrounded


def grounding_report(clause_set: ClauseSet, rate: float, ungrounded: List[Clause]) -> str:
    lines = [f"Grounding rate: {rate:.0%} of quoted spans located in the source."]
    if ungrounded:
        lines.append("Clauses with NO verifiable quote (treat as unsupported):")
        lines += [f"  - {c.id} ({c.heading}): {c.spans[0].quote[:120]!r}" for c in ungrounded]
    else:
        lines.append("Every clause has at least one verified quote.")
    return "\n".join(lines)
