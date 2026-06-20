import pytest

from personal_rag.core.schema import Document
from personal_rag.ingest.chunker import build_chunks, split_text


def make_document(content: str = "abcdefghij") -> Document:
    return Document(
        doc_id="doc_1",
        source_path="notes.md",
        doc_type="md",
        content=content,
        content_hash="file-hash",
        metadata={"title": "notes.md", "heading": "Intro", "page": None},
    )


def test_split_text_has_overlap_and_makes_forward_progress():
    chunks = split_text("abcdefghij", chunk_size=6, overlap=2)

    assert chunks == ["abcdef", "efghij"]


def test_split_text_prefers_paragraph_boundary():
    chunks = split_text("first paragraph\n\nsecond paragraph", chunk_size=20, overlap=0)

    assert chunks == ["first paragraph", "second paragraph"]


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (10, -1), (10, 10), (10, 11)],
)
def test_split_text_rejects_invalid_settings(chunk_size, overlap):
    with pytest.raises(ValueError):
        split_text("text", chunk_size=chunk_size, overlap=overlap)


def test_build_chunks_produces_stable_ids_and_metadata():
    first = build_chunks(make_document(), chunk_size=6, overlap=2)
    second = build_chunks(make_document(), chunk_size=6, overlap=2)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert first[0].metadata["source_path"] == "notes.md"
    assert first[0].metadata["heading"] == "Intro"

