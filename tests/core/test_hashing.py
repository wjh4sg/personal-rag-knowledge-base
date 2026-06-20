from personal_rag.core.hashing import make_doc_id, sha256_text


def test_make_id_is_stable_and_path_separator_independent():
    assert make_doc_id(r"notes\rag.md") == make_doc_id("notes/rag.md")


def test_sha256_text_changes_with_content():
    assert sha256_text("first") != sha256_text("second")

