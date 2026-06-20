from personal_rag.core.schema import RetrievedChunk
from personal_rag.rag.fusion import rrf_fusion


def result(chunk_id: str, source: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=f"text {chunk_id}",
        score=1.0,
        source=source,
        metadata={"source_path": f"{chunk_id}.md"},
    )


def test_rrf_rewards_chunks_present_in_both_rankings():
    a = result("a", "vector")
    b_vector = result("b", "vector")
    b_bm25 = result("b", "bm25")
    c = result("c", "bm25")

    fused = rrf_fusion([a, b_vector], [b_bm25, c], top_k=3)

    assert fused[0].chunk_id == "b"
    assert fused[0].source == "fusion"
    assert len({item.chunk_id for item in fused}) == 3


def test_rrf_respects_top_k():
    fused = rrf_fusion(
        [result("a", "vector"), result("b", "vector")],
        [result("c", "bm25")],
        top_k=2,
    )

    assert len(fused) == 2

