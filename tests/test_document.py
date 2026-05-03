"""Tests for Document + Chunk + chunking."""
import pytest

from rag_reference import Document, Chunk, chunk_document


class TestDocument:
    def test_construct_minimal(self):
        d = Document(doc_id="d1", text="hello world")
        assert d.doc_id == "d1"
        assert d.text == "hello world"
        assert d.metadata == {}

    def test_empty_doc_id_rejected(self):
        with pytest.raises(ValueError):
            Document(doc_id="", text="x")

    def test_empty_text_rejected(self):
        with pytest.raises(ValueError):
            Document(doc_id="d1", text="")
        with pytest.raises(ValueError):
            Document(doc_id="d1", text="   ")


class TestChunkDocument:
    def test_short_doc_one_chunk(self):
        d = Document(doc_id="d", text="hello world this is a tiny doc")
        chunks = chunk_document(d, chunk_size=200, overlap=40)
        assert len(chunks) == 1
        assert chunks[0].doc_id == "d"

    def test_long_doc_multiple_chunks(self):
        words = " ".join(f"word{i}" for i in range(500))
        d = Document(doc_id="d", text=words)
        chunks = chunk_document(d, chunk_size=100, overlap=20)
        assert len(chunks) >= 5
        # Step = chunk_size - overlap = 80
        assert chunks[1].start == 80
        assert chunks[2].start == 160

    def test_overlap_preserves_boundary_words(self):
        words = " ".join(f"w{i}" for i in range(150))
        d = Document(doc_id="d", text=words)
        chunks = chunk_document(d, chunk_size=100, overlap=20)
        # Last 20 words of chunk[0] should appear in chunk[1]
        last_20_of_first = chunks[0].text.split()[-20:]
        first_20_of_second = chunks[1].text.split()[:20]
        assert last_20_of_first == first_20_of_second

    def test_invalid_chunk_size_raises(self):
        d = Document(doc_id="d", text="hello")
        with pytest.raises(ValueError):
            chunk_document(d, chunk_size=0, overlap=0)

    def test_invalid_overlap_raises(self):
        d = Document(doc_id="d", text="hello")
        with pytest.raises(ValueError):
            chunk_document(d, chunk_size=10, overlap=10)
        with pytest.raises(ValueError):
            chunk_document(d, chunk_size=10, overlap=-1)

    def test_chunk_ids_are_deterministic(self):
        d = Document(doc_id="d", text=" ".join(["x"] * 300))
        a = chunk_document(d, chunk_size=100, overlap=20)
        b = chunk_document(d, chunk_size=100, overlap=20)
        assert [c.chunk_id for c in a] == [c.chunk_id for c in b]

    def test_chunk_metadata_inherited(self):
        d = Document(doc_id="d", text="x y z", metadata={"src": "wiki"})
        chunks = chunk_document(d, chunk_size=10, overlap=2)
        assert chunks[0].metadata["src"] == "wiki"
