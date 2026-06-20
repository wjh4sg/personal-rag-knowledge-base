from __future__ import annotations

from personal_rag.core.schema import RetrievedChunk


def rrf_fusion(
    vector_results: list[RetrievedChunk],
    bm25_results: list[RetrievedChunk],
    top_k: int = 5,
    k: int = 60,
) -> list[RetrievedChunk]:
    score_map: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}
    first_seen: dict[str, int] = {}
    seen_index = 0

    for ranking in (vector_results, bm25_results):
        for rank, item in enumerate(ranking, start=1):
            score_map[item.chunk_id] = score_map.get(item.chunk_id, 0.0) + 1.0 / (
                k + rank
            )
            chunk_map.setdefault(item.chunk_id, item)
            if item.chunk_id not in first_seen:
                first_seen[item.chunk_id] = seen_index
                seen_index += 1

    ranked_ids = sorted(
        score_map,
        key=lambda chunk_id: (-score_map[chunk_id], first_seen[chunk_id]),
    )
    return [
        RetrievedChunk(
            chunk_id=chunk_map[chunk_id].chunk_id,
            text=chunk_map[chunk_id].text,
            score=score_map[chunk_id],
            source="fusion",
            metadata=chunk_map[chunk_id].metadata,
        )
        for chunk_id in ranked_ids[:top_k]
    ]

