from __future__ import annotations

from personal_rag.core.hashing import sha256_text
from personal_rag.core.schema import Chunk, Document


def _preferred_end(text: str, start: int, target: int) -> int:
    if target >= len(text):
        return len(text)
    minimum = start + max(1, (target - start) // 2)
    for separator in ("\n\n", "\n", "。", "！", "？", ". ", " "):
        boundary = text.rfind(separator, minimum, target + 1)
        if boundary != -1:
            return boundary + (len(separator) if separator not in ("\n\n", "\n", " ") else 0)
    return target


def split_text(text: str, chunk_size: int = 700, overlap: int = 120) -> list[str]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        target = min(len(text), start + chunk_size)
        end = _preferred_end(text, start, target)
        if end <= start:
            end = target
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        next_start = end - overlap
        start = next_start if next_start > start else end
    return chunks


def build_chunks(
    document: Document,
    chunk_size: int = 700,
    overlap: int = 120,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for index, text in enumerate(split_text(document.content, chunk_size, overlap)):
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}_chunk_{index:04d}",
                doc_id=document.doc_id,
                text=text,
                chunk_hash=sha256_text(text),
                metadata={**document.metadata, "source_path": document.source_path},
            )
        )
    return chunks

