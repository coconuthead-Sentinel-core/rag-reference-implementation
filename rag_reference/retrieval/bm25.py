"""BM25Retriever — pure-Python Okapi BM25 lexical retriever.

Standard hybrid RAG: combine dense (vector) similarity with sparse
(lexical) BM25. BM25 wins on rare named entities and specific code
identifiers that dense embeddings smooth over.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from ..vectorstore.base import VectorHit


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", text)]


@dataclass
class _BM25Doc:
    chunk_id: str
    text:     str
    tokens:   list[str]
    length:   int
    tf:       Counter
    metadata: dict


class BM25Retriever:
    """Reference Okapi BM25 implementation (k1=1.5, b=0.75).

    Add chunks via ``add()``; search via ``search(query, k=N)``.
    """
    name = "bm25_retriever"

    def __init__(self, *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: list[_BM25Doc] = []
        self._df: Counter = Counter()        # doc-frequency per token
        self._avgdl: float = 0.0

    def add(self, chunk_id: str, text: str,
            metadata: dict | None = None) -> None:
        tokens = _tokenize(text)
        tf = Counter(tokens)
        doc = _BM25Doc(
            chunk_id=chunk_id,
            text=text,
            tokens=tokens,
            length=len(tokens),
            tf=tf,
            metadata=dict(metadata or {}),
        )
        self._docs.append(doc)
        for term in set(tokens):
            self._df[term] += 1
        self._recompute_avgdl()

    def add_many(self, items: Iterable[tuple[str, str, dict | None]]) -> None:
        for cid, text, meta in items:
            self.add(cid, text, meta)

    def _recompute_avgdl(self) -> None:
        if not self._docs:
            self._avgdl = 0.0
            return
        self._avgdl = sum(d.length for d in self._docs) / len(self._docs)

    def _idf(self, term: str) -> float:
        n_docs = len(self._docs)
        df = self._df.get(term, 0)
        # Robertson-Spärck-Jones IDF + 1 to avoid negative
        return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: list[str], doc: _BM25Doc) -> float:
        if doc.length == 0 or self._avgdl == 0:
            return 0.0
        score = 0.0
        for term in query_tokens:
            tf = doc.tf.get(term, 0)
            if tf == 0:
                continue
            idf = self._idf(term)
            denom = tf + self.k1 * (1 - self.b + self.b * doc.length / self._avgdl)
            score += idf * (tf * (self.k1 + 1)) / denom
        return score

    def search(self, query: str, *, k: int = 5) -> list[VectorHit]:
        if k < 1 or not self._docs:
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        scored: list[tuple[float, _BM25Doc]] = []
        for d in self._docs:
            scored.append((self.score(q_tokens, d), d))
        scored.sort(key=lambda t: t[0], reverse=True)
        out: list[VectorHit] = []
        for s, d in scored[:k]:
            if s <= 0:
                continue
            out.append(VectorHit(
                chunk_id=d.chunk_id,
                score=s,
                text=d.text,
                metadata=dict(d.metadata),
            ))
        return out

    def __len__(self) -> int:
        return len(self._docs)
