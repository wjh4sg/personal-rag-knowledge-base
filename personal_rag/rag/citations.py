from __future__ import annotations

import re

from personal_rag.core.schema import Citation, RetrievedChunk


def extract_citation_ids(answer_text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\[(C\d+)\]", answer_text)))


def check_citations(
    answer_text: str,
    citation_map: dict[str, RetrievedChunk],
) -> list[Citation]:
    citations: list[Citation] = []
    for citation_id in extract_citation_ids(answer_text):
        chunk = citation_map.get(citation_id)
        if chunk is None:
            continue
        citations.append(
            Citation(
                citation_id=citation_id,
                chunk_id=chunk.chunk_id,
                source_path=str(chunk.metadata.get("source_path", "")),
                page=chunk.metadata.get("page"),
                heading=chunk.metadata.get("heading"),
            )
        )
    return citations

