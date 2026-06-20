from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ChunkSettings:
    size: int
    overlap: int


@dataclass(frozen=True)
class RetrievalSettings:
    top_k: int
    vector_top_k: int
    bm25_top_k: int


@dataclass(frozen=True)
class ProviderSettings:
    mode: str
    base_url: str
    embedding_model: str
    embedding_dimensions: int
    llm_model: str
    timeout_seconds: int
    fallback_to_mock: bool
    api_key: str | None


@dataclass(frozen=True)
class StorageSettings:
    base_dir: Path
    cache_dir: Path


@dataclass(frozen=True)
class AppConfig:
    chunk: ChunkSettings
    retrieval: RetrievalSettings
    provider: ProviderSettings
    storage: StorageSettings
    config_path: Path


def _required_section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid configuration section: {name}")
    return value


def _resolve_path(config_dir: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def load_config(path: str | Path = "config/config.yaml") -> AppConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")

    chunk = _required_section(raw, "chunk")
    retrieval = _required_section(raw, "retrieval")
    provider = _required_section(raw, "provider")
    storage = _required_section(raw, "storage")

    chunk_size = int(chunk["size"])
    overlap = int(chunk["overlap"])
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Chunk settings require size > 0 and 0 <= overlap < size")

    return AppConfig(
        chunk=ChunkSettings(size=chunk_size, overlap=overlap),
        retrieval=RetrievalSettings(
            top_k=int(retrieval["top_k"]),
            vector_top_k=int(retrieval["vector_top_k"]),
            bm25_top_k=int(retrieval["bm25_top_k"]),
        ),
        provider=ProviderSettings(
            mode=str(os.getenv("RAG_PROVIDER_MODE", provider["mode"])).lower(),
            base_url=str(os.getenv("RAG_BASE_URL", provider["base_url"])).rstrip("/"),
            embedding_model=str(
                os.getenv("RAG_EMBEDDING_MODEL", provider["embedding_model"])
            ),
            embedding_dimensions=int(provider["embedding_dimensions"]),
            llm_model=str(os.getenv("RAG_MODEL", provider["llm_model"])),
            timeout_seconds=int(provider["timeout_seconds"]),
            fallback_to_mock=bool(provider.get("fallback_to_mock", True)),
            api_key=os.getenv("MODELSCOPE_API_KEY") or os.getenv("RAG_API_KEY"),
        ),
        storage=StorageSettings(
            base_dir=_resolve_path(config_path.parent, str(storage["base_dir"])),
            cache_dir=_resolve_path(config_path.parent, str(storage["cache_dir"])),
        ),
        config_path=config_path,
    )

