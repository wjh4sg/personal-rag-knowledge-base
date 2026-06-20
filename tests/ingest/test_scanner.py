from personal_rag.core.hashing import compute_file_hash
from personal_rag.ingest.scanner import scan_files


def test_scan_files_classifies_added_modified_deleted_and_unchanged(tmp_path):
    added = tmp_path / "added.md"
    added.write_text("new", encoding="utf-8")
    changed = tmp_path / "changed.txt"
    changed.write_text("changed", encoding="utf-8")
    unchanged = tmp_path / "same.md"
    unchanged.write_text("same", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("ignored", encoding="utf-8")

    old_state = {
        "changed.txt": {"content_hash": "old"},
        "same.md": {"content_hash": compute_file_hash(unchanged)},
        "deleted.pdf": {"content_hash": "gone"},
    }

    result = scan_files(tmp_path, old_state)

    assert [path.name for path in result.added] == ["added.md"]
    assert [path.name for path in result.modified] == ["changed.txt"]
    assert [path.name for path in result.unchanged] == ["same.md"]
    assert result.deleted == ["deleted.pdf"]
    assert result.unsupported_count == 1


def test_scan_files_is_recursive_and_uses_posix_relative_paths(tmp_path):
    nested = tmp_path / "folder" / "note.md"
    nested.parent.mkdir()
    nested.write_text("nested", encoding="utf-8")

    result = scan_files(tmp_path, {})

    assert result.added == [nested]

