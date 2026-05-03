"""InMemoryVectorStore — exact cosine top-k over a list. O(N) per search."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .base import VectorHit


@dataclass
class _Row:
    chunk_id: str
    vector:   list[float]
    text:     str
    metadata: dict[str, Any] = field(default_factory=dict)


class InMemoryVectorStore:
    """Reference store. Suitable for tests, demos, small corpora (<=10K chunks).
    For production, swap in chroma / pgvector / qdrant via the same protocol.
    """
    name = "in_memory_vector_store"

    def __init__(self) -> None:
        self._rows: list[_Row] = []
        self._by_id: dict[str, _Row] = {}

    def upsert(self, chunk_id: str, vector: list[float],
               text: str, metadata: dict[str, Any] | None = None) -> None:
        if not chunk_id:
            raise ValueError("chunk_id required")
        if not vector:
            raise ValueError("vector required")
        existing = self._by_id.get(chunk_id)
        row = _Row(chunk_id=chunk_id, vector=list(vector),
                   text=text, metadata=dict(metadata or {}))
        if existing is None:
            self._rows.append(row)
        else:
            # Replace in place
            i = self._rows.index(existing)
            self._rows[i] = row
        self._by_id[chunk_id] = row

    def search(self, query_vector: list[float], *, k: int = 5
               ) -> list[VectorHit]:
        if k < 1:
            return []
        if not self._rows:
            return []
        scored: list[tuple[float, _Row]] = []
        for r in self._rows:
            scored.append((_cosine(query_vector, r.vector), r))
        scored.sort(key=lambda t: t[0], reverse=True)
        out: list[VectorHit] = []
        for score, row in scored[:k]:
            out.append(VectorHit(
                chunk_id=row.chunk_id,
                score=score,
                text=row.text,
                metadata=dict(row.metadata),
            ))
        return out

    def __len__(self) -> int:
        return len(self._rows)

    def clear(self) -> None:
        self._rows.clear()
        self._by_id.clear()


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vector dim mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))
