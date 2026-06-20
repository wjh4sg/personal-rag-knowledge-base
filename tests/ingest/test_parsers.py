from pypdf import PdfWriter

from personal_rag.ingest.parsers import parse_file


def test_parse_markdown_preserves_heading_per_document_section(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Intro\nA\n## Retrieval\nB", encoding="utf-8")

    docs = parse_file(path, tmp_path)

    assert [doc.content for doc in docs] == ["A", "B"]
    assert [doc.metadata["heading"] for doc in docs] == ["Intro", "Retrieval"]
    assert all(doc.source_path == "notes.md" for doc in docs)


def test_parse_text_returns_one_document(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("plain text", encoding="utf-8")

    docs = parse_file(path, tmp_path)

    assert len(docs) == 1
    assert docs[0].content == "plain text"
    assert docs[0].metadata["page"] is None


def test_parse_pdf_skips_empty_pages(tmp_path):
    path = tmp_path / "empty.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with path.open("wb") as target:
        writer.write(target)

    assert parse_file(path, tmp_path) == []


def test_parse_file_rejects_unsupported_type(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b", encoding="utf-8")

    try:
        parse_file(path, tmp_path)
    except ValueError as error:
        assert "Unsupported file type" in str(error)
    else:
        raise AssertionError("Expected unsupported file type to fail")

