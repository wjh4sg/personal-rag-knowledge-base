from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Document:
    doc_id: str
    source_path: str
    doc_type: str
    content: str
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    chunk_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Citation:
    citation_id: str
    chunk_id: str
    source_path: str
    page: str | int | None
    heading: str | None


@dataclass(frozen=True)
class Answer:
    question: str
    answer: str
    citations: list[Citation]
    used_chunks: list[str]
    generation_mode: str = "mock"


@dataclass(frozen=True)
class ScanResult:
    added: list[Path]
    modified: list[Path]
    deleted: list[str]
    unchanged: list[Path]
    unsupported_count: int = 0


@dataclass(frozen=True)
class IndexReport:
    scanned: int
    added: int
    modified: int
    deleted: int
    unchanged: int
    document_count: int
    chunk_count: int
    embedding_cache_hits: int = 0
    skipped_pdf_pages: int = 0


@dataclass(frozen=True)
class EvaluationReport:
    sample_count: int
    hit_at_1: float
    hit_at_3: float
    mrr: float
    citation_coverage: float
    answers_without_citations: int

