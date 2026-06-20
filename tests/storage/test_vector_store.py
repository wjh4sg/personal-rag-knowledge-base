from personal_rag.core.schema import Chunk
from personal_rag.storage.vector_store import VectorStore


def make_chunk() -> Chunk:
    return Chunk(
        chunk_id="chunk_1",
        doc_id="doc_1",
        text="RRF 融合两路检索排名。",
        chunk_hash="hash_1",
        metadata={
            "source_path": "notes.md",
            "heading": "混合检索",
            "page": None,
        },
    )


def test_vector_store_upserts_queries_and_deletes(tmp_path):
    store = VectorStore(tmp_path / "chroma", "chunks")
    chunk = make_chunk()

    store.upsert([chunk], [[1.0, 0.0]])

    results = store.query([1.0, 0.0], top_k=1)
    assert results[0].chunk_id == chunk.chunk_id
    assert results[0].source == "vector"
    assert results[0].metadata["source_path"] == "notes.md"

    store.delete([chunk.chunk_id])

    assert store.query([1.0, 0.0], top_k=1) == []


def test_vector_store_reports_existence_after_upsert(tmp_path):
    store = VectorStore(tmp_path / "chroma", "chunks")

    assert not store.exists()
    store.upsert([make_chunk()], [[1.0, 0.0]])

    assert store.exists()

