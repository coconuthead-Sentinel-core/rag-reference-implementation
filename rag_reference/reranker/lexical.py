"""LexicalReranker — dependency-free reranker that boosts hits whose
text contains exact query tokens.

Production stacks swap in a cross-encoder. For tests + demos this is
deterministic and fast.
"""
from __future__ import annotations

import re

from ..vectorstore.base import VectorHit


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[A-Za-z0-9_]+", text)}


class LexicalReranker:
    """Boost candidates by exact-token overlap with the query."""
    name = "lexical_reranker"

    def __init__(self, *, boost_per_match: float = 0.05):
        self.boost_per_match = float(boost_per_match)

    def rerank(self, query: str, hits: list[VectorHit]) -> list[VectorHit]:
        q_tokens = _tokens(query)
        if not q_tokens:
            return list(hits)
        adjusted: list[VectorHit] = []
        for h in hits:
            overlap = len(q_tokens & _tokens(h.text))
            new_score = h.score + overlap * self.boost_per_match
            adjusted.append(VectorHit(
                chunk_id=h.chunk_id,
                score=new_score,
                text=h.text,
                metadata={**h.metadata, "rerank_overlap": overlap},
            ))
        adjusted.sort(key=lambda x: x.score, reverse=True)
        return adjusted
