from __future__ import annotations

import shutil
from pathlib import Path

import chromadb

from personal_rag.core.metadata import normalize_metadata
from personal_rag.core.schema import Chunk, RetrievedChunk


class VectorStore:
    def __init__(self, directory: Path, collection_name: str = "chunks"):
        self.directory = directory
        self.collection_name = collection_name
        self._connect()

    def _connect(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.directory))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have the same length")
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[normalize_metadata(chunk.metadata) for chunk in chunks],
            embeddings=embeddings,
        )

    def delete(self, chunk_ids: list[str]) -> None:
        if chunk_ids:
            self.collection.delete(ids=chunk_ids)

    def query(self, embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        count = self.collection.count()
        if count == 0 or top_k <= 0:
            return []
        response = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        ids = (response.get("ids") or [[]])[0]
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                text=document or "",
                score=1.0 - float(distance),
                source="vector",
                metadata=dict(metadata or {}),
            )
            for chunk_id, document, metadata, distance in zip(
                ids,
                documents,
                metadatas,
                distances,
                strict=True,
            )
        ]

    def exists(self) -> bool:
        return self.collection.count() > 0

    def count(self) -> int:
        return self.collection.count()

    def close(self) -> None:
        self.client.close()

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except ValueError:
            pass
        self.close()
        del self.collection
        del self.client
        shutil.rmtree(self.directory, ignore_errors=False)
        self._connect()

