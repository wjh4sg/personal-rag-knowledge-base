from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from personal_rag.core.hashing import compute_file_hash, make_file_id
from personal_rag.core.schema import Chunk, IndexReport
from personal_rag.ingest.chunker import build_chunks
from personal_rag.ingest.parsers import parse_file
from personal_rag.ingest.scanner import scan_files
from personal_rag.storage.bm25_store import BM25Store
from personal_rag.storage.chunk_store import ChunkStore
from personal_rag.storage.doc_store import DocStore
from personal_rag.storage.embedding_cache import EmbeddingCache
from personal_rag.storage.stats_store import StatsStore
from personal_rag.storage.vector_store import VectorStore


class Indexer:
    def __init__(
        self,
        *,
        doc_store: DocStore,
        chunk_store: ChunkStore,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        embedding_cache: EmbeddingCache,
        stats_store: StatsStore,
        embedding_client: Any,
        chunk_size: int = 700,
        overlap: int = 120,
    ):
        self.doc_store = doc_store
        self.chunk_store = chunk_store
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.embedding_cache = embedding_cache
        self.stats_store = stats_store
        self.embedding_client = embedding_client
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _cache_key(self, chunk: Chunk) -> str:
        return self.embedding_cache.key(
            self.embedding_client.provider_name,
            self.embedding_client.model_name,
            self.embedding_client.dimensions,
            chunk.chunk_hash,
        )

    def _embed_chunks(self, chunks: list[Chunk]) -> tuple[list[list[float]], int]:
        embeddings: list[list[float] | None] = [None] * len(chunks)
        missing_indices: list[int] = []
        cache_hits = 0
        for index, chunk in enumerate(chunks):
            cached = self.embedding_cache.get(self._cache_key(chunk))
            if cached is None:
                missing_indices.append(index)
            else:
                embeddings[index] = cached
                cache_hits += 1

        if missing_indices:
            generated = self.embedding_client.embed(
                [chunks[index].text for index in missing_indices]
            )
            if len(generated) != len(missing_indices):
                raise ValueError("Embedding provider returned an unexpected vector count")
            for index, vector in zip(missing_indices, generated, strict=True):
                embeddings[index] = vector
                self.embedding_cache.set(self._cache_key(chunks[index]), vector)

        if any(vector is None for vector in embeddings):
            raise RuntimeError("Failed to produce embeddings for every chunk")
        return [vector for vector in embeddings if vector is not None], cache_hits

    def index(self, docs_path: Path) -> IndexReport:
        docs_path = docs_path.resolve()
        old_state = self.doc_store.load()
        scan = scan_files(docs_path, old_state)
        old_chunks = self.chunk_store.load_all()

        removed_paths = set(scan.deleted)
        removed_paths.update(
            path.relative_to(docs_path).as_posix() for path in scan.modified
        )
        removed_doc_ids = {
            doc_id
            for relative_path in removed_paths
            for doc_id in old_state.get(relative_path, {}).get("doc_ids", [])
        }
        removed_chunk_ids = [
            chunk.chunk_id for chunk in old_chunks if chunk.doc_id in removed_doc_ids
        ]
        next_chunks = [
            chunk for chunk in old_chunks if chunk.doc_id not in removed_doc_ids
        ]
        next_state = {
            relative_path: dict(info)
            for relative_path, info in old_state.items()
            if relative_path not in removed_paths
        }

        new_chunks: list[Chunk] = []
        skipped_pdf_pages = 0
        now = datetime.now(timezone.utc).astimezone().isoformat()
        for path in [*scan.added, *scan.modified]:
            relative_path = path.relative_to(docs_path).as_posix()
            documents = parse_file(path, docs_path)
            if path.suffix.lower() == ".pdf":
                skipped_pdf_pages += max(0, len(PdfReader(str(path)).pages) - len(documents))
            file_chunks = [
                chunk
                for document in documents
                for chunk in build_chunks(
                    document,
                    chunk_size=self.chunk_size,
                    overlap=self.overlap,
                )
            ]
            new_chunks.extend(file_chunks)
            next_chunks.extend(file_chunks)
            next_state[relative_path] = {
                "file_id": make_file_id(relative_path),
                "content_hash": compute_file_hash(path),
                "doc_ids": [document.doc_id for document in documents],
                "updated_at": now,
            }

        embeddings, cache_hits = self._embed_chunks(new_chunks)
        changed = bool(scan.added or scan.modified or scan.deleted)
        needs_initial_publish = not (
            self.doc_store.exists()
            and self.chunk_store.exists()
            and self.bm25_store.exists()
        )

        if removed_chunk_ids:
            self.vector_store.delete(removed_chunk_ids)
        if new_chunks:
            self.vector_store.upsert(new_chunks, embeddings)
        if changed or needs_initial_publish:
            self.bm25_store.build(next_chunks)
            self.bm25_store.save()
            self.chunk_store.save_all(next_chunks)
            self.doc_store.save(next_state)

        self.stats_store.save(
            {
                "last_indexed_at": now,
                "docs_root": docs_path.as_posix(),
                "embedding_mode": self.embedding_client.provider_name,
                "embedding_model": self.embedding_client.model_name,
            }
        )
        return IndexReport(
            scanned=len(scan.added) + len(scan.modified) + len(scan.unchanged),
            added=len(scan.added),
            modified=len(scan.modified),
            deleted=len(scan.deleted),
            unchanged=len(scan.unchanged),
            document_count=sum(
                len(info.get("doc_ids", [])) for info in next_state.values()
            ),
            chunk_count=len(next_chunks),
            unsupported_count=scan.unsupported_count,
            embedding_cache_hits=cache_hits,
            skipped_pdf_pages=skipped_pdf_pages,
        )
