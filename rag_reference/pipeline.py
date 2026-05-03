"""RagPipeline — end-to-end orchestrator.

Wires Document -> Chunk -> Embedding -> VectorStore -> HybridRetriever
-> Reranker -> Generator with citation tracking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .citation import Citation
from .document import Chunk, Document, chunk_document
from .embeddings.base import EmbeddingProvider
from .generator.base import Generator
from .reranker.base import Reranker
from .retrieval.bm25 import BM25Retriever
from .retrieval.hybrid import HybridRetriever
from .vectorstore.base import VectorHit, VectorStore


@dataclass
class RagAnswer:
    """Final pipeline output — answer plus full citation chain."""
    query:     str
    answer:    str
    citations: list[Citation] = field(default_factory=list)
    metadata:  dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query":     self.query,
            "answer":    self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "metadata":  dict(self.metadata),
        }


@dataclass
class RagPipeline:
    """End-to-end RAG pipeline.

    Compose the four pluggable layers + ingest documents + ask
    questions:

        from rag_reference import (
            RagPipeline, HashEmbedding, InMemoryVectorStore,
            BM25Retriever, LexicalReranker, TemplateGenerator, Document,
        )

        pipe = RagPipeline(
            embedder=HashEmbedding(dim=128),
            vectorstore=InMemoryVectorStore(),
            bm25=BM25Retriever(),
            reranker=LexicalReranker(),
            generator=TemplateGenerator(),
        )

        pipe.ingest([Document("doc-1", "Python is a programming language ...")])
        ans = pipe.ask("what is python?")
        print(ans.answer)
    """
    embedder:    EmbeddingProvider
    vectorstore: VectorStore
    bm25:        BM25Retriever
    reranker:    Reranker
    generator:   Generator

    chunk_size:           int = 200
    chunk_overlap:        int = 40
    candidates_per_layer: int = 20
    top_k:                int = 5

    def ingest(self, documents: list[Document]) -> int:
        """Chunk + embed + index every document. Returns total chunks added."""
        added = 0
        for doc in documents:
            chunks = chunk_document(
                doc, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
            if not chunks:
                continue
            vectors = self.embedder.embed_batch([c.text for c in chunks])
            for c, v in zip(chunks, vectors):
                self.vectorstore.upsert(
                    chunk_id=c.chunk_id,
                    vector=v,
                    text=c.text,
                    metadata={**c.metadata, "doc_id": c.doc_id,
                              "start": c.start, "end": c.end},
                )
                self.bm25.add(c.chunk_id, c.text,
                              metadata={"doc_id": c.doc_id})
                added += 1
        return added

    def retrieve(self, query: str) -> list[VectorHit]:
        """Run hybrid retrieval + rerank, return top-K hits."""
        hybrid = HybridRetriever(
            embedder=self.embedder,
            vectorstore=self.vectorstore,
            bm25=self.bm25,
        )
        candidates = hybrid.search(
            query, k=self.candidates_per_layer,
            candidates_per_retriever=self.candidates_per_layer,
        )
        reranked = self.reranker.rerank(query, candidates)
        return reranked[: self.top_k]

    def ask(self, query: str) -> RagAnswer:
        """Full pipeline: retrieve -> rerank -> generate -> citations."""
        hits = self.retrieve(query)
        answer_text = self.generator.generate(query, hits)
        citations = self._build_citations(hits)
        return RagAnswer(
            query=query,
            answer=answer_text,
            citations=citations,
            metadata={
                "embedder":      self.embedder.name,
                "vectorstore":   self.vectorstore.name,
                "reranker":      self.reranker.name,
                "generator":     self.generator.name,
                "retrieved":     len(hits),
            },
        )

    def _build_citations(self, hits: list[VectorHit]) -> list[Citation]:
        out: list[Citation] = []
        for i, h in enumerate(hits, start=1):
            doc_id = str(h.metadata.get("doc_id", ""))
            snippet = h.text[:200]
            if len(h.text) > 200:
                snippet += "…"
            out.append(Citation(
                rank=i,
                chunk_id=h.chunk_id,
                doc_id=doc_id,
                score=h.score,
                snippet=snippet,
                metadata=dict(h.metadata),
            ))
        return out
