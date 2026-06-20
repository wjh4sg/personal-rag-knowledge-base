from __future__ import annotations

import pickle
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from personal_rag.core.schema import Chunk, RetrievedChunk
from personal_rag.storage.atomic import atomic_write_bytes


def tokenize(text: str) -> list[str]:
    return [token.strip().lower() for token in jieba.lcut(text) if token.strip()]


class BM25Store:
    def __init__(self, path: Path):
        self.path = path
        self.chunks: list[Chunk] = []
        self.corpus_tokens: list[list[str]] = []
        self.index: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        self.corpus_tokens = [tokenize(chunk.text) for chunk in chunks]
        self.index = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self.index is None or top_k <= 0:
            return []
        scores = self.index.get_scores(tokenize(query))
        ranked = sorted(
            enumerate(scores),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )
        results: list[RetrievedChunk] = []
        for index, score in ranked:
            if float(score) <= 0:
                continue
            chunk = self.chunks[index]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=float(score),
                    source="bm25",
                    metadata=chunk.metadata,
                )
            )
            if len(results) >= top_k:
                break
        return results

    def save(self) -> None:
        payload = {
            "chunks": self.chunks,
            "corpus_tokens": self.corpus_tokens,
        }
        atomic_write_bytes(self.path, pickle.dumps(payload))

    def load(self) -> None:
        if not self.path.exists():
            self.build([])
            return
        payload = pickle.loads(self.path.read_bytes())
        self.chunks = payload["chunks"]
        self.corpus_tokens = payload["corpus_tokens"]
        self.index = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def exists(self) -> bool:
        return self.path.exists()

