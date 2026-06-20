from datetime import datetime, timezone

from personal_rag.core.schema import Chunk
from personal_rag.storage.chunk_store import ChunkStore
from personal_rag.storage.doc_store import DocStore
from personal_rag.storage.embedding_cache import EmbeddingCache
from personal_rag.storage.stats_store import StatsStore


def make_chunk(chunk_id: str, doc_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        chunk_hash=f"hash-{chunk_id}",
        metadata={"source_path": f"{doc_id}.md", "page": None, "heading": None},
    )


def test_doc_store_round_trips_file_mapping(tmp_path):
    store = DocStore(tmp_path / "docs.json")
    state = {"notes.md": {"content_hash": "abc", "doc_ids": ["doc_1"]}}

    store.save(state)

    assert store.load()["notes.md"]["doc_ids"] == ["doc_1"]


def test_missing_doc_store_loads_empty_mapping(tmp_path):
    assert DocStore(tmp_path / "docs.json").load() == {}


def test_chunk_store_replaces_chunks_for_a_document(tmp_path):
    store = ChunkStore(tmp_path / "chunks.jsonl")
    old_a = make_chunk("old_a", "doc_a", "old")
    keep = make_chunk("keep", "doc_b", "keep")
    new_a = make_chunk("new_a", "doc_a", "new")
    store.save_all([old_a, keep])

    store.replace_doc_chunks("doc_a", [new_a])

    assert [chunk.chunk_id for chunk in store.load_all()] == ["keep", "new_a"]


def test_chunk_store_load_map_is_keyed_by_chunk_id(tmp_path):
    store = ChunkStore(tmp_path / "chunks.jsonl")
    chunk = make_chunk("chunk_1", "doc_1", "text")
    store.save_all([chunk])

    assert store.load_map() == {"chunk_1": chunk}


def test_embedding_cache_key_changes_with_model(tmp_path):
    cache = EmbeddingCache(tmp_path)

    assert cache.key("provider", "model-a", 3, "hash") != cache.key(
        "provider", "model-b", 3, "hash"
    )


def test_embedding_cache_round_trips_vector(tmp_path):
    cache = EmbeddingCache(tmp_path)
    key = cache.key("provider", "model", 3, "hash")

    assert cache.get(key) is None
    cache.set(key, [0.1, 0.2, 0.3])

    assert cache.get(key) == [0.1, 0.2, 0.3]


def test_stats_store_round_trips_index_metadata(tmp_path):
    store = StatsStore(tmp_path / "stats.json")
    indexed_at = datetime(2026, 6, 21, tzinfo=timezone.utc).isoformat()
    stats = {
        "last_indexed_at": indexed_at,
        "docs_root": "examples/docs",
        "embedding_mode": "mock",
        "embedding_model": "mock-hash-embedding",
    }

    store.save(stats)

    assert store.load() == stats

