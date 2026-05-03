"""Reranker protocol."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..vectorstore.base import VectorHit


@runtime_checkable
class Reranker(Protocol):
    """Re-orders the top-K candidates from a retriever to maximize
    relevance to the user query. Production stacks plug in
    cross-encoders (Cohere Rerank v3, BGE reranker, etc.).
    """
    name: str

    def rerank(self, query: str, hits: list[VectorHit]) -> list[VectorHit]: ...
