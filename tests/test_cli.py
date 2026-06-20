import yaml

from personal_rag.cli import main
from personal_rag.core.schema import Chunk
from personal_rag.storage.bm25_store import BM25Store
from personal_rag.storage.chunk_store import ChunkStore


def write_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "chunk": {"size": 100, "overlap": 20},
                "retrieval": {
                    "top_k": 5,
                    "vector_top_k": 10,
                    "bm25_top_k": 10,
                },
                "provider": {
                    "mode": "mock",
                    "base_url": "https://api-inference.modelscope.cn/v1",
                    "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
                    "embedding_dimensions": 32,
                    "llm_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                    "timeout_seconds": 10,
                    "fallback_to_mock": True,
                },
                "storage": {
                    "base_dir": "storage",
                    "cache_dir": "cache",
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_cli_exposes_all_commands(capsys):
    exit_code = main(["--help"])
    output = capsys.readouterr().out

    assert exit_code == 0
    for command in ["index", "search", "ask", "stats", "eval", "rebuild"]:
        assert command in output


def test_cli_returns_missing_index_code_for_search(tmp_path, capsys):
    config = write_config(tmp_path)

    exit_code = main(["--config", str(config), "search", "query"])
    output = capsys.readouterr().err

    assert exit_code == 3
    assert "rag index" in output


def test_stats_works_before_first_index(tmp_path, capsys):
    config = write_config(tmp_path)

    exit_code = main(["--config", str(config), "stats"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "文档数：0" in output
    assert "Chunk 数：0" in output


def test_cli_returns_argument_error_for_missing_command(capsys):
    assert main([]) == 2
    assert "usage:" in capsys.readouterr().err.lower()


def test_cli_rejects_incomplete_index_when_chroma_is_empty(tmp_path, capsys):
    config = write_config(tmp_path)
    chunk = Chunk(
        chunk_id="chunk_1",
        doc_id="doc_1",
        text="incomplete index",
        chunk_hash="hash_1",
        metadata={"source_path": "notes.md", "page": None, "heading": None},
    )
    base = tmp_path / "storage"
    ChunkStore(base / "chunks.jsonl").save_all([chunk])
    bm25 = BM25Store(base / "bm25.pkl")
    bm25.build([chunk])
    bm25.save()

    exit_code = main(["--config", str(config), "search", "query"])
    error = capsys.readouterr().err

    assert exit_code == 3
    assert "索引不存在或不完整" in error
    assert "rag rebuild" in error


def test_rebuild_validates_docs_directory_before_reset(tmp_path, capsys):
    config = write_config(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.txt").write_text("existing knowledge", encoding="utf-8")
    assert main(["--config", str(config), "index", str(docs)]) == 0
    capsys.readouterr()
    manifest = tmp_path / "storage" / "docs.json"
    before = manifest.read_text(encoding="utf-8")

    exit_code = main(
        ["--config", str(config), "rebuild", str(tmp_path / "missing-docs")]
    )

    assert exit_code == 2
    assert manifest.read_text(encoding="utf-8") == before

