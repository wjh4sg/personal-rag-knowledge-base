from __future__ import annotations

import hashlib
from pathlib import Path


def normalize_relative_path(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_file_id(relative_path: str | Path) -> str:
    return f"file_{sha256_text(normalize_relative_path(relative_path))[:16]}"


def make_doc_id(relative_path: str | Path, page: int | None = None) -> str:
    base = f"doc_{sha256_text(normalize_relative_path(relative_path))[:16]}"
    return f"{base}_p{page}" if page is not None else base

