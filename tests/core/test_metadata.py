from personal_rag.core.metadata import normalize_metadata


def test_normalize_metadata_replaces_none_and_stringifies_complex_values():
    assert normalize_metadata({"page": None, "tags": ["rag"]}) == {
        "page": "",
        "tags": "['rag']",
    }


def test_normalize_metadata_keeps_supported_scalar_types():
    metadata = {"title": "notes", "page": 3, "score": 0.5, "active": True}

    assert normalize_metadata(metadata) == metadata

