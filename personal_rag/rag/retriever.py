from __future__ import annotations

from typing import Any

from personal_rag.core.schema import RetrievedChunk
from personal_rag.rag.fusion import rrf_fusion


class HybridRetriever:
    def __init__(
        self,
        *,
        embedding_client: Any,
        vector_store: Any,
        bm25_store: Any,
        vector_top_k: int = 10,
        bm25_top_k: int = 10,
        top_k: int = 5,
    ):
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        query_embedding = self.embedding_client.embed([query])[0]
        vector_results = self.vector_store.query(
            query_embedding,
            top_k=self.vector_top_k,
        )
        bm25_results = self.bm25_store.search(query, top_k=self.bm25_top_k)
        return rrf_fusion(
            vector_results,
            bm25_results,
            top_k=top_k if top_k is not None else self.top_k,
        )

