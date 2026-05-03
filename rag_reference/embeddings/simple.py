"""HashEmbedding — dependency-free deterministic embedding for tests + demos.

Maps tokens into a fixed-dimension vector via the classic
"hashing trick" (Weinberger et al. 2009). The same text always produces
the same vector. Real similarity properties are weaker than dense
neural embeddings but strong enough that semantically related texts
score higher than unrelated ones, which is sufficient for the
end-to-end pipeline + the test suite.

Production deployments swap this out for OpenAI/Cohere/SBERT.
"""
from __future__ import annotations

import hashlib
import math
import re


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z]+", text)]


class HashEmbedding:
    """Hashing-trick embedder."""
    name = "hash_embedding"

    def __init__(self, dim: int = 128, ngrams: tuple[int, ...] = (1, 2)):
        if dim < 8:
            raise ValueError("dim must be >= 8")
        self.dim = dim
        self.ngrams = tuple(ngrams)

    def _features(self, text: str) -> list[str]:
        tokens = _tokenize(text)
        out: list[str] = []
        for n in self.ngrams:
            for i in range(0, max(0, len(tokens) - n + 1)):
                out.append(" ".join(tokens[i:i + n]))
        return out

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        feats = self._features(text)
        if not feats:
            return vec
        for f in feats:
            h = hashlib.md5(f.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if (h[4] & 1) else -1.0
            vec[idx] += sign
        # L2 normalize so cosine == dot product
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]
