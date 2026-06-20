from personal_rag.core.schema import Chunk
from personal_rag.storage.bm25_store import BM25Store


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=f"doc_{chunk_id}",
        text=text,
        chunk_hash=f"hash_{chunk_id}",
        metadata={"source_path": f"{chunk_id}.md", "heading": None, "page": None},
    )


def test_bm25_store_ranks_exact_chinese_term(tmp_path):
    store = BM25Store(tmp_path / "bm25.pkl")
    store.build(
        [
            make_chunk("rerank", "精排可以重新判断候选片段与问题的相关性。"),
            make_chunk("vector", "向量检索解决语义相似问题。"),
            make_chunk("cache", "缓存可以减少重复计算。"),
        ]
    )

    results = store.search("为什么需要精排", top_k=1)

    assert results[0].chunk_id == "rerank"
    assert results[0].source == "bm25"


def test_bm25_store_persists_and_loads(tmp_path):
    path = tmp_path / "bm25.pkl"
    original = BM25Store(path)
    original.build(
        [
            make_chunk("rrf", "RRF 使用排名进行融合。"),
            make_chunk("vector", "向量召回语义内容。"),
            make_chunk("cache", "缓存复用向量。"),
        ]
    )
    original.save()

    loaded = BM25Store(path)
    loaded.load()

    assert loaded.search("RRF 排名", top_k=1)[0].chunk_id == "rrf"
    assert loaded.exists()


def test_bm25_empty_index_returns_no_results(tmp_path):
    store = BM25Store(tmp_path / "bm25.pkl")
    store.build([])

    assert store.search("query", top_k=5) == []

