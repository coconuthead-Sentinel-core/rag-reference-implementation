"""Tests for TemplateGenerator."""
from rag_reference import Generator, TemplateGenerator, VectorHit


class TestTemplateGenerator:
    def test_implements_protocol(self):
        g = TemplateGenerator()
        assert isinstance(g, Generator)

    def test_no_hits_returns_polite_no_context(self):
        g = TemplateGenerator()
        out = g.generate("what is python?", [])
        assert "no relevant context" in out.lower()

    def test_renders_question(self):
        g = TemplateGenerator()
        out = g.generate("what is python?",
                         [VectorHit(chunk_id="a", score=0.9,
                                    text="Python is a language.")])
        assert "what is python?" in out

    def test_renders_each_hit_with_index(self):
        g = TemplateGenerator()
        hits = [
            VectorHit(chunk_id="a", score=0.9, text="alpha"),
            VectorHit(chunk_id="b", score=0.8, text="bravo"),
        ]
        out = g.generate("q", hits)
        assert "[1]" in out and "alpha" in out
        assert "[2]" in out and "bravo" in out

    def test_includes_citations(self):
        g = TemplateGenerator()
        hits = [VectorHit(chunk_id="my-chunk-42", score=0.91, text="x")]
        out = g.generate("q", hits)
        assert "my-chunk-42" in out

    def test_max_chunks_caps_output(self):
        g = TemplateGenerator(max_chunks=2)
        hits = [VectorHit(chunk_id=str(i), score=1.0 - i * 0.1, text=f"t{i}")
                for i in range(5)]
        out = g.generate("q", hits)
        assert "[2]" in out
        assert "[3]" not in out

    def test_long_chunks_get_truncated(self):
        g = TemplateGenerator(max_chars_per_chunk=20)
        long_text = "x" * 500
        hits = [VectorHit(chunk_id="a", score=0.9, text=long_text)]
        out = g.generate("q", hits)
        assert "…" in out
