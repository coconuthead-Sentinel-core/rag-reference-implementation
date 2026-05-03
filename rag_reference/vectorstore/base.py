"""VectorStore protocol + VectorHit."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class VectorHit:
    """A single search result."""
    chunk_id: str
    score:    float
    text:     str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class VectorStore(Protocol):
    """Holds (chunk_id, vector, text, metadata) tuples and supports
    cosine top-k search.

    Production adapters wrap chromadb / pinecone / pgvector / qdrant.
    """
    name: str

    def upsert(self, chunk_id: str, vector: list[float],
               text: str, metadata: dict[str, Any] | None = None) -> None: ...

    def search(self, query_vector: list[float], *, k: int = 5
               ) -> list[VectorHit]: ...

    def __len__(self) -> int: ...

    def clear(self) -> None: ...
