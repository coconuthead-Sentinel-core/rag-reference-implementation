"""End-to-end pipeline tests."""
from rag_reference import (
    Document,
    RagPipeline,
    HashEmbedding,
    InMemoryVectorStore,
    BM25Retriever,
    LexicalReranker,
    TemplateGenerator,
)


def _build_pipeline():
    return RagPipeline(
        embedder=HashEmbedding(dim=128),
        vectorstore=InMemoryVectorStore(),
        bm25=BM25Retriever(),
        reranker=LexicalReranker(),
        generator=TemplateGenerator(),
        chunk_size=30,
        chunk_overlap=5,
        candidates_per_layer=10,
        top_k=3,
    )


CORPUS = [
    Document("py-intro",
             "Python is a high-level interpreted programming language. "
             "It is widely used in data science, web development, and AI. "
             "Its syntax emphasizes readability."),
    Document("fastapi-doc",
             "FastAPI is a modern Python web framework for building APIs. "
             "It uses Pydantic for data validation and ASGI under the hood."),
    Document("rust-vs-c",
             "Rust is a systems programming language focused on memory "
             "safety without a garbage collector."),
    Document("vector-db-overview",
             "A vector database stores dense embeddings and supports "
             "approximate nearest neighbor search for retrieval-augmented "
             "generation pipelines."),
]


class TestRagPipeline:
    def test_ingest_returns_chunk_count(self):
        p = _build_pipeline()
        n = p.ingest(CORPUS)
        assert n >= len(CORPUS)        # at least 1 chunk per doc
        assert len(p.vectorstore) == n
        assert len(p.bm25) == n

    def test_ask_returns_answer_object(self):
        p = _build_pipeline()
        p.ingest(CORPUS)
        ans = p.ask("what is python?")
        assert ans.query == "what is python?"
        assert ans.answer
        assert isinstance(ans.citations, list)

    def test_citations_carry_doc_id(self):
        p = _build_pipeline()
        p.ingest(CORPUS)
        ans = p.ask("python web framework FastAPI")
        assert len(ans.citations) >= 1
        # At least one citation should point at fastapi-doc
        doc_ids = {c.doc_id for c in ans.citations}
        assert "fastapi-doc" in doc_ids or "py-intro" in doc_ids

    def test_metadata_tracks_components(self):
        p = _build_pipeline()
        p.ingest(CORPUS)
        ans = p.ask("python")
        assert ans.metadata["embedder"] == "hash_embedding"
        assert ans.metadata["vectorstore"] == "in_memory_vector_store"
        assert ans.metadata["reranker"] == "lexical_reranker"
        assert ans.metadata["generator"] == "template_generator"

    def test_retrieve_returns_top_k_max(self):
        p = _build_pipeline()
        p.ingest(CORPUS)
        hits = p.retrieve("python")
        assert len(hits) <= p.top_k

    def test_to_dict_round_trip(self):
        p = _build_pipeline()
        p.ingest(CORPUS)
        ans = p.ask("python")
        d = ans.to_dict()
        assert d["query"] == "python"
        assert "answer" in d
        assert isinstance(d["citations"], list)

    def test_query_unrelated_to_corpus_does_not_crash(self):
        p = _build_pipeline()
        p.ingest(CORPUS)
        ans = p.ask("zzz nonsense rocketships")
        # Pipeline must respond — even if hits are weak
        assert ans.query == "zzz nonsense rocketships"
        assert isinstance(ans.answer, str)

    def test_ingest_then_query_finds_correct_doc(self):
        p = _build_pipeline()
        p.ingest(CORPUS)
        ans = p.ask("vector database approximate nearest neighbor")
        # vector-db-overview should be in top citations
        doc_ids = {c.doc_id for c in ans.citations}
        assert "vector-db-overview" in doc_ids
