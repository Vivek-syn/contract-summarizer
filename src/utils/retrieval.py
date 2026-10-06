"""BM25 retrieval, dependency-free.

Used to pick the clauses relevant to each playbook rule, which is what keeps
validation rule-major and each call small.

Lexical rather than semantic on purpose: playbook rules are written in the same
legal terms of art that appear in the contract, so exact-term matching works
well and there is no second provider to configure. Swap ``Index`` for a vector
store if that stops being true - only ``search()`` is used.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

_TOKEN = re.compile(r"[a-z0-9']+")
_STOP = {
    "the", "a", "an", "of", "to", "and", "or", "in", "for", "on", "by", "with",
    "shall", "any", "such", "this", "that", "be", "is", "are", "as", "at", "it",
    "not", "no", "which", "may", "must", "will",
}


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


@dataclass
class Document:
    """One retrievable unit."""

    id: str
    text: str
    meta: Dict[str, str] = field(default_factory=dict)


class Index:
    """Okapi BM25 over a fixed document set."""

    def __init__(self, documents: Sequence[Document], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = list(documents)
        self.k1, self.b = k1, b
        tokens = [tokenize(d.text) for d in self.documents]
        self._tf = [Counter(t) for t in tokens]
        self._lengths = [len(t) for t in tokens]
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0

        df: Counter = Counter()
        for token_list in tokens:
            df.update(set(token_list))
        n = max(len(self.documents), 1)
        self._idf = {
            term: math.log(1 + (n - count + 0.5) / (count + 0.5)) for term, count in df.items()
        }

    def __len__(self) -> int:
        return len(self.documents)

    def search(self, query: str, top_k: int = 8) -> List[Document]:
        """The ``top_k`` best-matching documents, highest score first."""
        terms = tokenize(query)
        if not terms or not self.documents:
            return []

        scored: List[tuple] = []
        for index, tf in enumerate(self._tf):
            score = 0.0
            for term in terms:
                freq = tf.get(term)
                if not freq:
                    continue
                idf = self._idf.get(term, 0.0)
                denominator = freq + self.k1 * (
                    1 - self.b + self.b * self._lengths[index] / (self._avg_len or 1)
                )
                score += idf * (freq * (self.k1 + 1)) / denominator
            if score > 0:
                scored.append((score, index))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [self.documents[i] for _, i in scored[:top_k]]


def index_clauses(clauses: Sequence) -> Index:
    """Build an index over extracted clauses.

    Heading, category and summary all carry signal a rule query may hit, so all
    three are indexed alongside the quoted text.
    """
    return Index(
        [
            Document(
                id=c.id,
                text=f"{c.heading} {c.category} {c.summary} {c.quote}",
                meta={"page": str(c.page)},
            )
            for c in clauses
        ]
    )


def index_texts(entries: Sequence[tuple]) -> Index:
    """Build an index from ``(id, text)`` pairs."""
    return Index([Document(id=key, text=body) for key, body in entries])
