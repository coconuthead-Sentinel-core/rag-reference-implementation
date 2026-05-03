"""Tests for InMemoryVectorStore + cosine retrieval."""
import pytest

from rag_reference import (
    HashEmbedding,
    InMemoryVectorStore,
    VectorStore,
    VectorHit,
)


class TestInMemoryVectorStore:
    def test_implements_protocol(self):
        s = InMemoryVectorStore()
        assert isinstance(s, VectorStore)

    def test_starts_empty(self):
        s = InMemoryVectorStore()
        assert len(s) == 0
        assert s.search([0.1] * 8, k=5) == []

    def test_upsert_grows(self):
        s = InMemoryVectorStore()
        s.upsert("a", [1.0, 0.0, 0.0], "first")
        assert len(s) == 1

    def test_upsert_replaces_in_place(self):
        s = InMemoryVectorStore()
        s.upsert("a", [1.0, 0.0, 0.0], "first")
        s.upsert("a", [0.0, 1.0, 0.0], "first replaced")
        assert len(s) == 1
        hits = s.search([0.0, 1.0, 0.0], k=1)
        assert hits[0].text == "first replaced"

    def test_empty_chunk_id_rejected(self):
        s = InMemoryVectorStore()
        with pytest.raises(ValueError):
            s.upsert("", [1.0, 0.0], "x")

    def test_empty_vector_rejected(self):
        s = InMemoryVectorStore()
        with pytest.raises(ValueError):
            s.upsert("a", [], "x")

    def test_search_returns_top_k(self):
        s = InMemoryVectorStore()
        s.upsert("a", [1.0, 0.0, 0.0], "x")
        s.upsert("b", [0.0, 1.0, 0.0], "y")
        s.upsert("c", [0.0, 0.0, 1.0], "z")
        hits = s.search([1.0, 0.0, 0.0], k=2)
        assert len(hits) == 2
        assert hits[0].chunk_id == "a"
        assert hits[0].score > hits[1].score

    def test_search_dim_mismatch_raises(self):
        s = InMemoryVectorStore()
        s.upsert("a", [1.0, 0.0, 0.0], "x")
        with pytest.raises(ValueError):
            s.search([1.0, 0.0], k=1)

    def test_search_returns_vectorhit_objects(self):
        s = InMemoryVectorStore()
        s.upsert("a", [1.0, 0.0], "abc")
        hits = s.search([1.0, 0.0], k=1)
        assert isinstance(hits[0], VectorHit)
        assert hits[0].text == "abc"

    def test_clear(self):
        s = InMemoryVectorStore()
        s.upsert("a", [1.0, 0.0], "x")
        s.clear()
        assert len(s) == 0

    def test_end_to_end_with_hash_embedding(self):
        e = HashEmbedding(dim=128)
        s = InMemoryVectorStore()
        for cid, txt in [("a", "python programming"),
                         ("b", "python is great"),
                         ("c", "the cat sat on the mat")]:
            s.upsert(cid, e.embed(txt), txt)
        hits = s.search(e.embed("python language"), k=2)
        # Top hits should be the python-related ones
        ids = [h.chunk_id for h in hits]
        assert "a" in ids or "b" in ids
        assert hits[0].chunk_id != "c"
