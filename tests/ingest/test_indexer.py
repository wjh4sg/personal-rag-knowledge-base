from types import SimpleNamespace

import pytest

from personal_rag.ingest.indexer import Indexer
from personal_rag.providers.embeddings import MockEmbeddingClient
from personal_rag.storage.bm25_store import BM25Store
from personal_rag.storage.chunk_store import ChunkStore
from personal_rag.storage.doc_store import DocStore
from personal_rag.storage.embedding_cache import EmbeddingCache
from personal_rag.storage.stats_store import StatsStore
from personal_rag.storage.vector_store import VectorStore


class RecordingEmbeddingClient(MockEmbeddingClient):
    def __init__(self):
        super().__init__(dimensions=32)
        self.calls = 0
        self.text_count = 0

    def embed(self, texts):
        self.calls += 1
        self.text_count += len(texts)
        return super().embed(texts)


@pytest.fixture
def index_env(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    storage = tmp_path / "storage"
    embedding = RecordingEmbeddingClient()
    doc_store = DocStore(storage / "docs.json")
    chunk_store = ChunkStore(storage / "chunks.jsonl")
    vector_store = VectorStore(storage / "chroma", "test_chunks")
    bm25_store = BM25Store(storage / "bm25.pkl")
    indexer = Indexer(
        doc_store=doc_store,
        chunk_store=chunk_store,
        vector_store=vector_store,
        bm25_store=bm25_store,
        embedding_cache=EmbeddingCache(tmp_path / "cache"),
        stats_store=StatsStore(storage / "stats.json"),
        embedding_client=embedding,
        chunk_size=30,
        overlap=5,
    )
    return SimpleNamespace(
        docs=docs,
        storage=storage,
        embedding=embedding,
        doc_store=doc_store,
        chunk_store=chunk_store,
        vector_store=vector_store,
        bm25_store=bm25_store,
        indexer=indexer,
    )


def test_second_unchanged_index_skips_embedding(index_env):
    (index_env.docs / "notes.txt").write_text("增量索引使用内容哈希判断变化。", encoding="utf-8")

    first = index_env.indexer.index(index_env.docs)
    calls_after_first = index_env.embedding.calls
    second = index_env.indexer.index(index_env.docs)

    assert first.added == 1
    assert second.unchanged == 1
    assert index_env.embedding.calls == calls_after_first
    assert first.chunk_count == second.chunk_count


def test_modified_file_replaces_only_its_chunks(index_env):
    changed = index_env.docs / "changed.txt"
    changed.write_text("old text about indexing", encoding="utf-8")
    untouched = index_env.docs / "untouched.txt"
    untouched.write_text("stable text about retrieval", encoding="utf-8")
    index_env.indexer.index(index_env.docs)
    untouched_doc_id = index_env.doc_store.load()["untouched.txt"]["doc_ids"][0]

    changed.write_text("new text about content hash", encoding="utf-8")
    report = index_env.indexer.index(index_env.docs)
    chunks = index_env.chunk_store.load_all()

    assert report.modified == 1
    assert all(chunk.text != "old text about indexing" for chunk in chunks)
    assert any(chunk.doc_id == untouched_doc_id for chunk in chunks)


def test_deleted_file_removes_manifest_chunks_and_vectors(index_env):
    deleted = index_env.docs / "deleted.txt"
    deleted.write_text("this file will be deleted", encoding="utf-8")
    index_env.indexer.index(index_env.docs)
    old_chunk_id = index_env.chunk_store.load_all()[0].chunk_id

    deleted.unlink()
    report = index_env.indexer.index(index_env.docs)

    assert report.deleted == 1
    assert index_env.doc_store.load() == {}
    assert index_env.chunk_store.load_all() == []
    assert index_env.vector_store.count() == 0
    assert old_chunk_id not in index_env.chunk_store.load_map()


def test_embedding_cache_reuses_vector_when_content_returns(index_env):
    path = index_env.docs / "notes.txt"
    path.write_text("cache me", encoding="utf-8")
    index_env.indexer.index(index_env.docs)
    first_text_count = index_env.embedding.text_count
    path.write_text("different content", encoding="utf-8")
    index_env.indexer.index(index_env.docs)
    path.write_text("cache me", encoding="utf-8")

    report = index_env.indexer.index(index_env.docs)

    assert report.embedding_cache_hits == 1
    assert index_env.embedding.text_count == first_text_count + 1

