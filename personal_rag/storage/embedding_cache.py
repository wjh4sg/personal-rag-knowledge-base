from __future__ import annotations

import json
from pathlib import Path

from personal_rag.core.hashing import sha256_text
from personal_rag.storage.atomic import atomic_write_json


class EmbeddingCache:
    def __init__(self, directory: Path):
        self.directory = directory

    @staticmethod
    def key(provider: str, model: str, dimensions: int, chunk_hash: str) -> str:
        identity = f"{provider}\n{model}\n{dimensions}\n{chunk_hash}"
        return sha256_text(identity)

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> list[float] | None:
        path = self._path(key)
        if not path.exists():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError(f"Invalid embedding cache entry: {path}")
        return [float(item) for item in value]

    def set(self, key: str, vector: list[float]) -> None:
        atomic_write_json(self._path(key), vector)

