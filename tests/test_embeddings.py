"""Tests for embedding providers."""
import math

import pytest

from rag_reference import EmbeddingProvider, HashEmbedding


class TestHashEmbedding:
    def test_implements_protocol(self):
        e = HashEmbedding()
        assert isinstance(e, EmbeddingProvider)

    def test_dim_default_128(self):
        e = HashEmbedding()
        assert e.dim == 128

    def test_dim_custom(self):
        e = HashEmbedding(dim=64)
        assert e.dim == 64

    def test_invalid_dim_rejected(self):
        with pytest.raises(ValueError):
            HashEmbedding(dim=4)

    def test_embed_returns_correct_length(self):
        e = HashEmbedding(dim=64)
        v = e.embed("hello world")
        assert len(v) == 64

    def test_embed_l2_normalized(self):
        e = HashEmbedding(dim=64)
        v = e.embed("hello world this is a test")
        norm = math.sqrt(sum(x * x for x in v))
        assert abs(norm - 1.0) < 1e-6

    def test_embed_deterministic(self):
        e = HashEmbedding(dim=32)
        v1 = e.embed("hello world")
        v2 = e.embed("hello world")
        assert v1 == v2

    def test_embed_distinguishes_different_text(self):
        e = HashEmbedding(dim=64)
        v1 = e.embed("python is a programming language")
        v2 = e.embed("the cat sat on the mat")
        # Cosine similarity should be < 1
        dot = sum(a * b for a, b in zip(v1, v2))
        assert dot < 0.99

    def test_embed_empty_returns_zero_vector(self):
        e = HashEmbedding(dim=32)
        v = e.embed("")
        assert v == [0.0] * 32

    def test_embed_batch_returns_n_vectors(self):
        e = HashEmbedding(dim=32)
        vs = e.embed_batch(["a", "b", "c"])
        assert len(vs) == 3
        for v in vs:
            assert len(v) == 32

    def test_similar_text_higher_similarity(self):
        """Texts sharing tokens should be more similar than unrelated ones."""
        e = HashEmbedding(dim=256, ngrams=(1, 2))
        v_python1 = e.embed("python programming language tutorial")
        v_python2 = e.embed("python programming language guide")
        v_unrelated = e.embed("the cat sat on the colorful mat")

        sim_close = sum(a * b for a, b in zip(v_python1, v_python2))
        sim_far = sum(a * b for a, b in zip(v_python1, v_unrelated))
        assert sim_close > sim_far
