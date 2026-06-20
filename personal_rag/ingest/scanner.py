from __future__ import annotations

from pathlib import Path
from typing import Any

from personal_rag.core.hashing import compute_file_hash
from personal_rag.core.schema import ScanResult

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


def scan_files(docs_path: Path, old_state: dict[str, dict[str, Any]]) -> ScanResult:
    docs_path = docs_path.resolve()
    if not docs_path.is_dir():
        raise FileNotFoundError(f"Document directory does not exist: {docs_path}")

    current_paths: set[str] = set()
    added: list[Path] = []
    modified: list[Path] = []
    unchanged: list[Path] = []
    unsupported_count = 0

    for path in sorted(docs_path.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            unsupported_count += 1
            continue

        relative_path = path.relative_to(docs_path).as_posix()
        current_paths.add(relative_path)
        current_hash = compute_file_hash(path)
        previous = old_state.get(relative_path)
        if previous is None:
            added.append(path)
        elif previous.get("content_hash") != current_hash:
            modified.append(path)
        else:
            unchanged.append(path)

    deleted = sorted(relative_path for relative_path in old_state if relative_path not in current_paths)
    return ScanResult(
        added=added,
        modified=modified,
        deleted=deleted,
        unchanged=unchanged,
        unsupported_count=unsupported_count,
    )

