"""Citation tracking — preserves the chain from final answer back to
source chunks back to source documents.

Production RAG without citation tracking is unauditable. Every claim
in the final answer should be traceable to a specific chunk_id, which
is in turn traceable to a specific Document.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Citation:
    """One citation entry in the chain answer -> chunk -> document."""
    rank:     int
    chunk_id: str
    doc_id:   str
    score:    float
    snippet:  str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "rank":     self.rank,
            "chunk_id": self.chunk_id,
            "doc_id":   self.doc_id,
            "score":    self.score,
            "snippet":  self.snippet,
            "metadata": dict(self.metadata),
        }
