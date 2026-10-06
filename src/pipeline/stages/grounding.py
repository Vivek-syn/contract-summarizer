"""Grounding: turn a quoted clause into coordinates on the page.

No model call. This is the one stage that cannot hallucinate, which is why
everything downstream is anchored to it.

The chain, and every link can fail:

    quote -> exact match in OCR text -> char span
          -> (on miss) fuzzy match    -> char span + score
          -> map span to word indices -> group words by line
          -> union each line's boxes  -> one box per line

One box per line, not one per clause: words wrap, so a clause spanning three
lines is three rectangles. A clause whose quote cannot be located at all is
marked ``missing`` and surfaced to the reviewer rather than quietly dropped.
"""

from __future__ import annotations

import difflib
import re
from typing import Dict, List, Optional, Tuple

from src.core.logging.structured import get_logger
from src.schemas.documents import (
    AnalysisResult,
    BoundingBox,
    GroundedClause,
    GroundedRegion,
    GroundingResult,
    OCRPage,
    OCRResult,
    OCRWord,
)

logger = get_logger(__name__)

_WS = re.compile(r"\s+")
_FUZZY_THRESHOLD = 0.82
# Words whose vertical centres fall within this fraction of page height are one line.
_LINE_TOLERANCE = 0.008


def _canon(text: str) -> str:
    """Whitespace- and quote-insensitive form, for matching only."""
    return _WS.sub(
        " ",
        text.lower()
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-"),
    ).strip()


def _canon_map(text: str) -> Tuple[str, List[int]]:
    """Canonical text plus, per canonical char, its offset in the original.

    Matching happens on the canonical string; boxes are addressed by original
    offsets, so the mapping between them has to be exact.
    """
    out: List[str] = []
    offsets: List[int] = []
    previous_space = True
    for index, char in enumerate(text):
        lowered = (
            char.lower()
            .replace("’", "'")
            .replace("‘", "'")
            .replace("“", '"')
            .replace("”", '"')
            .replace("–", "-")
            .replace("—", "-")
        )
        if lowered.isspace():
            if previous_space:
                continue
            out.append(" ")
            offsets.append(index)
            previous_space = True
        else:
            out.append(lowered)
            offsets.append(index)
            previous_space = False
    while out and out[-1] == " ":
        out.pop()
        offsets.pop()
    return "".join(out), offsets


def _fuzzy_locate(needle: str, haystack: str) -> Optional[Tuple[int, int, float]]:
    """Best approximate location of ``needle`` in ``haystack``.

    OCR is imperfect, so an exact miss does not prove the model hallucinated.
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
    score = difflib.SequenceMatcher(None, haystack[start:end], needle, autojunk=False).ratio()
    return (start, end, score) if score >= _FUZZY_THRESHOLD else None


def _words_in_span(page: OCRPage, start: int, end: int) -> List[OCRWord]:
    """Words overlapping ``[start, end)`` in the page's own text."""
    return [w for w in page.words if w.start < end and w.end > start]


def _group_into_lines(words: List[OCRWord]) -> List[List[OCRWord]]:
    """Group words into visual lines by vertical position."""
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (round(w.box.top, 4), w.box.left))
    lines: List[List[OCRWord]] = [[ordered[0]]]
    for word in ordered[1:]:
        previous = lines[-1][-1]
        same_line = abs(word.box.top - previous.box.top) <= _LINE_TOLERANCE
        lines[-1].append(word) if same_line else lines.append([word])
    return lines


def _boxes_for_words(words: List[OCRWord]) -> List[BoundingBox]:
    """One union box per visual line."""
    boxes: List[BoundingBox] = []
    for line in _group_into_lines(words):
        box = line[0].box
        for word in line[1:]:
            box = box.union(word.box)
        boxes.append(box)
    return boxes


def _locate_in_page(page: OCRPage, quote: str) -> Optional[Tuple[int, int, str, float]]:
    """Find ``quote`` in one page. Returns original-text offsets."""
    canon_page, offsets = _canon_map(page.text)
    canon_quote = _canon(quote)
    if not canon_quote or not canon_page:
        return None

    exact = canon_page.find(canon_quote)
    if exact != -1:
        start = offsets[exact]
        end_index = min(exact + len(canon_quote) - 1, len(offsets) - 1)
        return start, offsets[end_index] + 1, "exact", 1.0

    hit = _fuzzy_locate(canon_quote, canon_page)
    if hit:
        c_start, c_end, score = hit
        start = offsets[min(c_start, len(offsets) - 1)]
        end_index = min(max(c_end - 1, 0), len(offsets) - 1)
        return start, offsets[end_index] + 1, "fuzzy", score
    return None


def ground_clause(clause, ocr: OCRResult) -> GroundedClause:
    """Locate one clause's quote and compute its regions.

    The clause's own ``page`` is tried first, then every other page - models do
    mislabel pages, and the located text is more trustworthy than the label.
    """
    grounded = GroundedClause(clause_id=clause.id)
    pages = {p.page: p for p in ocr.pages if p.ok}
    if not pages:
        return grounded

    order = [clause.page] + [n for n in sorted(pages) if n != clause.page]
    for page_number in order:
        page = pages.get(page_number)
        if page is None:
            continue
        located = _locate_in_page(page, clause.quote)
        if located is None:
            continue

        start, end, match, score = located
        words = _words_in_span(page, start, end)
        boxes = _boxes_for_words(words)

        grounded.match = match  # type: ignore[assignment]
        grounded.score = score
        grounded.char_start = start
        grounded.char_end = end
        grounded.regions = [GroundedRegion(page=page_number, boxes=boxes)] if boxes else []
        grounded.min_confidence = min((w.confidence for w in words), default=1.0)

        if page_number != clause.page:
            logger.info(
                "clause %s was labelled page %d but found on page %d",
                clause.id, clause.page, page_number,
            )
        return grounded

    return grounded


def run(analysis: AnalysisResult, ocr: OCRResult) -> GroundingResult:
    """Ground every clause against the OCR text.

    Returns:
        A :class:`GroundingResult`. Clauses are also updated in place with their
        resolved character offsets.
    """
    grounded: List[GroundedClause] = []
    for clause in analysis.clauses:
        result = ground_clause(clause, ocr)
        clause.char_start = result.char_start
        clause.char_end = result.char_end
        if result.regions:
            clause.page = result.regions[0].page
        grounded.append(result)

    out = GroundingResult(clauses=grounded)
    logger.info(
        "grounding: %.0f%% located (%d exact, %d fuzzy, %d missing)",
        out.rate * 100,
        sum(1 for c in grounded if c.match == "exact"),
        sum(1 for c in grounded if c.match == "fuzzy"),
        sum(1 for c in grounded if c.match == "missing"),
    )
    return out
