"""The RAG box: a dependency-free BM25 index over the document and the rulebook.

Anthropic has no embeddings endpoint, so rather than drag in a second provider
this uses lexical BM25. For contracts that is a reasonable default - the queries
are legal terms of art that appear verbatim in the text. Swap ``Index`` for a
vector store if you need semantic recall; the interface is just ``search()``.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .config import SETTINGS

_TOKEN = re.compile(r"[a-z0-9']+")
_STOP = {
    "the", "a", "an", "of", "to", "and", "or", "in", "for", "on", "by", "with",
    "shall", "any", "such", "this", "that", "be", "is", "are", "as", "at", "it",
}


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


@dataclass
class Chunk:
    id: str
    text: str
    page: int
    offset: int
    meta: Dict[str, str]


def chunk_document(doc, *, size: int | None = None, overlap: int | None = None) -> List[Chunk]:
    """Sliding-window chunks that snap to paragraph boundaries where possible."""
    size = size or SETTINGS.chunk_chars
    overlap = overlap or SETTINGS.chunk_overlap
    text = doc.text
    chunks: List[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind("\n\n", start + size // 2, end)
            if boundary != -1:
                end = boundary
        body = text[start:end].strip()
        if body:
            chunks.append(
                Chunk(
                    id=f"c{len(chunks):04d}",
                    text=body,
                    page=doc.page_at(start),
                    offset=start,
                    meta={"source": doc.path.name},
                )
            )
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


class Index:
    """Okapi BM25 over a fixed chunk set."""

    def __init__(self, chunks: Sequence[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self._tokens = [tokenize(c.text) for c in self.chunks]
        self._lengths = [len(t) for t in self._tokens]
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        self._tf = [Counter(t) for t in self._tokens]

        df: Counter[str] = Counter()
        for tokens in self._tokens:
            df.update(set(tokens))
        n = max(len(self.chunks), 1)
        self._idf = {
            term: math.log(1 + (n - count + 0.5) / (count + 0.5)) for term, count in df.items()
        }

    def search(self, query: str, top_k: int | None = None) -> List[Chunk]:
        top_k = top_k or SETTINGS.top_k
        terms = tokenize(query)
        if not terms or not self.chunks:
            return []
        scores: List[tuple[float, int]] = []
        for i, tf in enumerate(self._tf):
            score = 0.0
            for term in terms:
                freq = tf.get(term)
                if not freq:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = freq + self.k1 * (1 - self.b + self.b * self._lengths[i] / (self._avg_len or 1))
                score += idf * (freq * (self.k1 + 1)) / denom
            if score > 0:
                scores.append((score, i))
        scores.sort(reverse=True)
        return [self.chunks[i] for _, i in scores[:top_k]]

    def render(self, chunks: Iterable[Chunk]) -> str:
        parts = [
            f"<chunk id=\"{c.id}\" page=\"{c.page}\">\n{c.text}\n</chunk>" for c in chunks
        ]
        return "\n\n".join(parts) if parts else "(no matching passages)"


def index_of_texts(entries: Sequence[tuple[str, str]]) -> Index:
    """Build an index from ``(id, text)`` pairs - used for the playbook/rulebook."""
    chunks = [Chunk(id=key, text=body, page=0, offset=0, meta={"source": "playbook"}) for key, body in entries]
    return Index(chunks)
