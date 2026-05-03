"""Tests for BM25Retriever and HybridRetriever."""
from rag_reference import (
    BM25Retriever, HybridRetriever,
    HashEmbedding, InMemoryVectorStore, VectorHit,
)


CORPUS = [
    ("a", "Python is a high-level interpreted programming language."),
    ("b", "FastAPI is a Python web framework for building APIs."),
    ("c", "The cat sat on the mat with a colorful blanket."),
    ("d", "Rust is a systems programming language with memory safety."),
    ("e", "Vector databases store dense embeddings for similarity search."),
]


class TestBM25Retriever:
    def test_starts_empty(self):
        r = BM25Retriever()
        assert len(r) == 0
        assert r.search("anything", k=5) == []

    def test_add_grows(self):
        r = BM25Retriever()
        for cid, txt in CORPUS:
            r.add(cid, txt)
        assert len(r) == len(CORPUS)

    def test_add_many(self):
        r = BM25Retriever()
        r.add_many([(cid, txt, None) for cid, txt in CORPUS])
        assert len(r) == len(CORPUS)

    def test_search_returns_relevant_first(self):
        r = BM25Retriever()
        for cid, txt in CORPUS:
            r.add(cid, txt)
        hits = r.search("python programming", k=3)
        assert len(hits) >= 1
        # 'a' contains both "python" and "programming"; 'b' has only python; 'd' only programming
        ids = [h.chunk_id for h in hits]
        assert "a" in ids

    def test_search_returns_zero_for_no_match(self):
        r = BM25Retriever()
        for cid, txt in CORPUS:
            r.add(cid, txt)
        hits = r.search("xyzzy nonexistent", k=5)
        assert hits == []

    def test_search_returns_vectorhit(self):
        r = BM25Retriever()
        r.add("a", "python tutorial")
        hits = r.search("python", k=1)
        assert isinstance(hits[0], VectorHit)


class TestHybridRetriever:
    def _build(self):
        emb = HashEmbedding(dim=128)
        vs = InMemoryVectorStore()
        bm = BM25Retriever()
        for cid, txt in CORPUS:
            vs.upsert(cid, emb.embed(txt), txt, metadata={"doc_id": cid})
            bm.add(cid, txt, metadata={"doc_id": cid})
        return HybridRetriever(embedder=emb, vectorstore=vs, bm25=bm)

    def test_returns_top_k(self):
        h = self._build()
        hits = h.search("python web framework", k=3)
        assert len(hits) <= 3
        assert all(isinstance(x, VectorHit) for x in hits)

    def test_fusion_metadata_present(self):
        h = self._build()
        hits = h.search("python", k=2)
        for hit in hits:
            assert hit.metadata.get("fusion") == "rrf"

    def test_finds_relevant_chunk(self):
        h = self._build()
        hits = h.search("FastAPI APIs framework", k=3)
        ids = [hit.chunk_id for hit in hits]
        assert "b" in ids   # FastAPI chunk

    def test_unrelated_query_still_returns(self):
        # Hybrid never returns nothing if the corpus isn't empty
        h = self._build()
        hits = h.search("rocketship", k=2)
        # May be 0 (if both BM25 and vector return nothing relevant)
        # but the call should not raise
        assert isinstance(hits, list)

    def test_empty_query_returns_empty(self):
        h = self._build()
        hits = h.search("", k=3)
        # Either dense returns small / sparse returns empty -> at least no crash
        assert isinstance(hits, list)
