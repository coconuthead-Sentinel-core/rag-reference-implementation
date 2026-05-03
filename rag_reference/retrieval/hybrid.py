"""HybridRetriever — fuse dense (vector) + sparse (BM25) results via RRF.

Reciprocal Rank Fusion (Cormack et al. 2009) is the de-facto fusion
algorithm in production RAG. Each retriever contributes a ranked list;
the fused score for a chunk is sum(1 / (rank + k_fuse)) across all
retrievers it appears in.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..vectorstore.base import VectorHit, VectorStore
from ..embeddings.base import EmbeddingProvider
from .bm25 import BM25Retriever


@dataclass
class HybridRetriever:
    """Combine a dense vector store with a BM25 lexical index using RRF."""
    embedder:    EmbeddingProvider
    vectorstore: VectorStore
    bm25:        BM25Retriever
    k_fuse:      int = 60                 # standard RRF constant

    name: str = "hybrid_retriever"

    def search(self, query: str, *, k: int = 5,
               candidates_per_retriever: int = 20) -> list[VectorHit]:
        """Return up to `k` chunks ranked by Reciprocal Rank Fusion."""
        if k < 1:
            return []
        q_vec = self.embedder.embed(query)
        dense_hits  = self.vectorstore.search(q_vec, k=candidates_per_retriever)
        sparse_hits = self.bm25.search(query, k=candidates_per_retriever)

        # RRF score per chunk_id
        fused: dict[str, float] = {}
        keep: dict[str, VectorHit] = {}

        for rank, hit in enumerate(dense_hits):
            fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + \
                                   1.0 / (rank + self.k_fuse)
            keep.setdefault(hit.chunk_id, hit)

        for rank, hit in enumerate(sparse_hits):
            fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + \
                                   1.0 / (rank + self.k_fuse)
            keep.setdefault(hit.chunk_id, hit)

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        out: list[VectorHit] = []
        for chunk_id, fused_score in ranked[:k]:
            base = keep[chunk_id]
            out.append(VectorHit(
                chunk_id=chunk_id,
                score=fused_score,                # RRF score
                text=base.text,
                metadata={**base.metadata, "fusion": "rrf"},
            ))
        return out
