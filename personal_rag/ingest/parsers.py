from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from personal_rag.core.hashing import compute_file_hash, make_doc_id
from personal_rag.core.schema import Document

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _relative_path(path: Path, docs_root: Path) -> str:
    return path.resolve().relative_to(docs_root.resolve()).as_posix()


def parse_markdown(path: Path, docs_root: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="replace")
    relative_path = _relative_path(path, docs_root)
    content_hash = compute_file_hash(path)
    base_id = make_doc_id(relative_path)
    sections: list[tuple[str | None, str]] = []
    heading: str | None = None
    lines: list[str] = []

    def publish() -> None:
        content = "\n".join(lines).strip()
        if content:
            sections.append((heading, content))

    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            publish()
            heading = match.group(2).strip()
            lines = []
        else:
            lines.append(line)
    publish()

    return [
        Document(
            doc_id=f"{base_id}_s{index:04d}",
            source_path=relative_path,
            doc_type="md",
            content=content,
            content_hash=content_hash,
            metadata={"title": path.name, "page": None, "heading": section_heading},
        )
        for index, (section_heading, content) in enumerate(sections)
    ]


def parse_text(path: Path, docs_root: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    relative_path = _relative_path(path, docs_root)
    return [
        Document(
            doc_id=make_doc_id(relative_path),
            source_path=relative_path,
            doc_type="txt",
            content=text,
            content_hash=compute_file_hash(path),
            metadata={"title": path.name, "page": None, "heading": None},
        )
    ]


def parse_pdf(path: Path, docs_root: Path) -> list[Document]:
    relative_path = _relative_path(path, docs_root)
    content_hash = compute_file_hash(path)
    documents: list[Document] = []
    for page_number, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        documents.append(
            Document(
                doc_id=make_doc_id(relative_path, page=page_number),
                source_path=relative_path,
                doc_type="pdf",
                content=text,
                content_hash=content_hash,
                metadata={"title": path.name, "page": page_number, "heading": None},
            )
        )
    return documents


def parse_file(path: Path, docs_root: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".md":
        return parse_markdown(path, docs_root)
    if suffix == ".txt":
        return parse_text(path, docs_root)
    if suffix == ".pdf":
        return parse_pdf(path, docs_root)
    raise ValueError(f"Unsupported file type: {suffix}")

