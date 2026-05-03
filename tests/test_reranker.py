"""Tests for LexicalReranker."""
from rag_reference import LexicalReranker, Reranker, VectorHit


class TestLexicalReranker:
    def test_implements_protocol(self):
        r = LexicalReranker()
        assert isinstance(r, Reranker)

    def test_empty_input(self):
        r = LexicalReranker()
        assert r.rerank("python", []) == []

    def test_empty_query_no_op(self):
        r = LexicalReranker()
        hits = [VectorHit(chunk_id="a", score=0.5, text="anything")]
        out = r.rerank("", hits)
        assert out[0].chunk_id == "a"

    def test_boosts_token_overlap(self):
        r = LexicalReranker(boost_per_match=0.1)
        hits = [
            VectorHit(chunk_id="a", score=0.50, text="python tutorial"),
            VectorHit(chunk_id="b", score=0.55, text="cat on the mat"),
        ]
        out = r.rerank("python programming", hits)
        # 'a' has 1 overlap (python) -> 0.50 + 0.10 = 0.60
        # 'b' has 0 overlaps -> stays at 0.55
        assert out[0].chunk_id == "a"
        assert out[0].score > out[1].score

    def test_metadata_records_overlap(self):
        r = LexicalReranker()
        hits = [VectorHit(chunk_id="a", score=0.5,
                          text="python rocks python")]
        out = r.rerank("python", hits)
        # set-based overlap, so {'python'} ∩ {'python','rocks'} = 1
        assert out[0].metadata["rerank_overlap"] == 1

    def test_preserves_chunk_id(self):
        r = LexicalReranker()
        hits = [VectorHit(chunk_id="zzz", score=0.5, text="anything")]
        out = r.rerank("python", hits)
        assert out[0].chunk_id == "zzz"
