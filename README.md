# RAG Reference Implementation v1.0

> **Pluggable Retrieval-Augmented Generation pipeline.**
> Embed → vector store → hybrid retrieval (dense + BM25 with RRF) →
> reranker → generator → citations. Five protocols, five reference
> backends, zero external API keys to run.

![Status](https://img.shields.io/badge/status-public-success)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-pytest-blue)

---

## What this is

The canonical reference implementation of the standard 6-stage RAG
pipeline:

```
Document  →  Chunk  →  Embedding  →  VectorStore
                                          │
                            +─────────────┴─────────────+
                            ▼                           ▼
                   Dense (cosine top-k)        Sparse (BM25 top-k)
                            └─────────────┬─────────────┘
                                          ▼
                          Reciprocal Rank Fusion (RRF)
                                          ▼
                                       Reranker
                                          ▼
                                       Generator
                                          ▼
                              Answer + Citations
```

Five pluggable protocols (`EmbeddingProvider`, `VectorStore`,
`Reranker`, `Generator`, plus the `BM25Retriever` class) ship with
dependency-free reference backends so the package runs end-to-end on a
fresh machine with **zero external API keys**. Real adapters
(OpenAI / Cohere / chromadb / pgvector / Anthropic / etc.) plug into
the same protocols.

## Install

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from rag_reference import (
    Document, RagPipeline,
    HashEmbedding, InMemoryVectorStore,
    BM25Retriever, LexicalReranker, TemplateGenerator,
)

pipe = RagPipeline(
    embedder    = HashEmbedding(dim=128),
    vectorstore = InMemoryVectorStore(),
    bm25        = BM25Retriever(),
    reranker    = LexicalReranker(),
    generator   = TemplateGenerator(),
    chunk_size  = 200,
    chunk_overlap = 40,
    top_k       = 5,
)

pipe.ingest([
    Document("py-intro",
             "Python is a high-level interpreted programming language ..."),
    Document("fastapi-doc",
             "FastAPI is a modern Python web framework for building APIs ..."),
])

answer = pipe.ask("what python web framework should I use?")
print(answer.answer)
for c in answer.citations:
    print(f"  [{c.rank}] {c.doc_id}/{c.chunk_id}  score={c.score:.3f}")
```

## Why the protocols matter

Every RAG production deployment ends up swapping at least one of:

| Layer | Common production swap |
|---|---|
| `EmbeddingProvider` | OpenAI `text-embedding-3-large`, Cohere v3, sentence-transformers |
| `VectorStore`       | chromadb, pgvector, qdrant, pinecone, weaviate |
| `BM25Retriever`     | Elasticsearch / OpenSearch sparse index |
| `Reranker`          | Cohere Rerank v3, BGE Reranker, cross-encoders |
| `Generator`         | OpenAI / Anthropic / local LLM |

This package gives you the *protocol* contracts so you can swap any
single layer without rewriting the pipeline.

## Hybrid retrieval — Reciprocal Rank Fusion (Cormack 2009)

`HybridRetriever` queries the dense vector store + the BM25 lexical
retriever in parallel, then fuses both rankings using RRF:

```
fused_score(chunk) = Σ over retrievers: 1 / (rank + k_fuse)
```

Default `k_fuse = 60` matches the original Cormack paper. RRF is the
de-facto fusion algorithm in production RAG because it doesn't require
score calibration between retrievers.

## Citation tracking

Every `RagAnswer` carries a `citations` list. Each `Citation` records:

- `rank` (1-based)
- `chunk_id`
- `doc_id`
- `score` (post-rerank)
- `snippet` (200-char preview)
- `metadata` (whatever the retriever passed through)

Production RAG without citations is unauditable — every claim should
trace back to a specific chunk.

## Testing

```bash
pytest -v
```

## Project structure

```
RAG Reference Implementation/
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
├── rag_reference/
│   ├── __init__.py
│   ├── document.py             ← Document + Chunk + chunk_document()
│   ├── citation.py             ← Citation
│   ├── pipeline.py             ← RagPipeline + RagAnswer
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── base.py             ← EmbeddingProvider protocol
│   │   └── simple.py           ← HashEmbedding
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   ├── base.py             ← VectorStore + VectorHit
│   │   └── inmemory.py         ← InMemoryVectorStore
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── bm25.py             ← BM25Retriever (Okapi BM25)
│   │   └── hybrid.py           ← HybridRetriever (RRF)
│   ├── reranker/
│   │   ├── __init__.py
│   │   ├── base.py             ← Reranker protocol
│   │   └── lexical.py          ← LexicalReranker
│   └── generator/
│       ├── __init__.py
│       ├── base.py             ← Generator protocol
│       └── template.py         ← TemplateGenerator
├── tests/
│   ├── test_document.py
│   ├── test_embeddings.py
│   ├── test_vectorstore.py
│   ├── test_retrieval.py
│   ├── test_reranker.py
│   ├── test_generator.py
│   ├── test_pipeline.py
│   └── test_citation.py
└── docs/
```

## License

MIT — see [`LICENSE`](LICENSE).

## Author

**Shannon Brian Kelly** — AI Orchestrator Architect.
Co-authored with Claude AI (Anthropic) under file-system-bound persona
protocol; co-creator role: **"Archivist of Wisdom"**.

Canon entry **#24** in the architect's portfolio.
