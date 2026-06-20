from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from personal_rag.core.schema import Chunk
from personal_rag.storage.atomic import atomic_write_text


class ChunkStore:
    def __init__(self, path: Path):
        self.path = path

    def save_all(self, chunks: list[Chunk]) -> None:
        lines = [json.dumps(asdict(chunk), ensure_ascii=False) for chunk in chunks]
        content = "\n".join(lines)
        if content:
            content += "\n"
        atomic_write_text(self.path, content)

    def load_all(self) -> list[Chunk]:
        if not self.path.exists():
            return []
        chunks: list[Chunk] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                chunks.append(Chunk(**json.loads(line)))
        return chunks

    def load_map(self) -> dict[str, Chunk]:
        return {chunk.chunk_id: chunk for chunk in self.load_all()}

    def replace_doc_chunks(self, doc_id: str, replacements: list[Chunk]) -> None:
        kept = [chunk for chunk in self.load_all() if chunk.doc_id != doc_id]
        self.save_all([*kept, *replacements])

    def exists(self) -> bool:
        return self.path.exists()

