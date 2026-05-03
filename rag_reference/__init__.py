"""
rag_reference — Retrieval-Augmented Generation Reference Implementation

Canon entry #24. Pluggable RAG pipeline with the standard 6-stage
flow:

  1. Document ingest + chunk
  2. Embedding (EmbeddingProvider protocol)
  3. Vector store (VectorStore protocol)
  4. Retrieval (semantic + lexical/BM25 hybrid)
  5. Reranker (Reranker protocol)
  6. Generation with citation tracking (Generator protocol)

All four protocols ship with dependency-free reference implementations
so the package runs end-to-end with no external API keys. Real
adapters (OpenAI, Cohere, Anthropic, etc.) plug into the same
protocols.
"""
from __future__ import annotations

from .document import Document, Chunk, chunk_document
from .embeddings.base import EmbeddingProvider
from .embeddings.simple import HashEmbedding
from .vectorstore.base import VectorStore, VectorHit
from .vectorstore.inmemory import InMemoryVectorStore
from .retrieval.bm25 import BM25Retriever
from .retrieval.hybrid import HybridRetriever
from .reranker.base import Reranker
from .reranker.lexical import LexicalReranker
from .generator.base import Generator
from .generator.template import TemplateGenerator
from .pipeline import RagPipeline, RagAnswer
from .citation import Citation

__version__ = "1.0.0"

__all__ = [
    "Document", "Chunk", "chunk_document",
    "EmbeddingProvider", "HashEmbedding",
    "VectorStore", "VectorHit", "InMemoryVectorStore",
    "BM25Retriever", "HybridRetriever",
    "Reranker", "LexicalReranker",
    "Generator", "TemplateGenerator",
    "RagPipeline", "RagAnswer",
    "Citation",
]
