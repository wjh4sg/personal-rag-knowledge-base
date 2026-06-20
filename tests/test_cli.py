import yaml

from personal_rag.cli import main


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
    for command in ["index", "search", "ask", "stats", "eval"]:
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

