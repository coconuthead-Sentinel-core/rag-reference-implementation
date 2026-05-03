"""EmbeddingProvider protocol."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Convert texts into fixed-dimension dense vectors.

    Implementations may wrap OpenAI, Cohere, sentence-transformers, etc.
    The reference implementation in ``simple.HashEmbedding`` is
    dependency-free and deterministic so the package runs end-to-end
    in any environment.
    """
    name: str
    dim:  int

    def embed(self, text: str) -> list[float]:
        """Return a `self.dim`-length vector for `text`."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch; default implementations may loop."""
        ...
