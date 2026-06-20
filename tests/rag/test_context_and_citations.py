from personal_rag.core.schema import RetrievedChunk
from personal_rag.rag.citations import check_citations, extract_citation_ids
from personal_rag.rag.context import build_context


def make_chunk(chunk_id: str = "chunk_1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text="系统使用内容 Hash 判断文件变化。",
        score=0.03,
        source="fusion",
        metadata={
            "source_path": "design.md",
            "page": None,
            "heading": "增量索引",
        },
    )


def test_context_assigns_stable_citation_identifiers():
    built = build_context("怎么做增量索引？", [make_chunk()])

    assert "[C1] 来源：design.md" in built.prompt
    assert built.citation_map == {"C1": make_chunk()}
    assert "资料不足" in built.prompt


def test_checker_filters_unknown_citations_and_deduplicates():
    chunk = make_chunk()

    citations = check_citations("事实。[C1][C99] 再次引用。[C1]", {"C1": chunk})

    assert [citation.citation_id for citation in citations] == ["C1"]
    assert citations[0].source_path == "design.md"
    assert citations[0].heading == "增量索引"


def test_extract_citation_ids_preserves_first_seen_order():
    assert extract_citation_ids("[C2] [C1] [C2]") == ["C2", "C1"]

