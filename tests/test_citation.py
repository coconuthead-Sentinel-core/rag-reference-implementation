"""Tests for Citation dataclass."""
from rag_reference import Citation


class TestCitation:
    def test_to_dict_keys(self):
        c = Citation(rank=1, chunk_id="c1", doc_id="d1",
                     score=0.95, snippet="hello",
                     metadata={"k": "v"})
        d = c.to_dict()
        for key in ("rank", "chunk_id", "doc_id", "score", "snippet", "metadata"):
            assert key in d

    def test_metadata_default_empty(self):
        c = Citation(rank=1, chunk_id="c1", doc_id="d1",
                     score=0.5, snippet="x")
        assert c.metadata == {}
