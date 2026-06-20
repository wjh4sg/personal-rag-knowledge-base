import shutil
from pathlib import Path

import yaml

from personal_rag.cli import main

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_config(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "chunk": {"size": 180, "overlap": 30},
                "retrieval": {
                    "top_k": 5,
                    "vector_top_k": 10,
                    "bm25_top_k": 10,
                },
                "provider": {
                    "mode": "mock",
                    "base_url": "https://api-inference.modelscope.cn/v1",
                    "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
                    "embedding_dimensions": 64,
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


def test_all_five_acceptance_commands(tmp_path, capsys):
    docs = tmp_path / "docs"
    shutil.copytree(PROJECT_ROOT / "examples" / "docs", docs)
    dataset = tmp_path / "dataset.json"
    shutil.copy2(PROJECT_ROOT / "eval" / "dataset.json", dataset)
    config = write_config(tmp_path / "config.yaml")
    common = ["--config", str(config)]

    assert main([*common, "index", str(docs)]) == 0
    index_output = capsys.readouterr().out
    assert "Chroma" in index_output
    assert "BM25" in index_output

    assert main([*common, "search", "RAG 为什么需要 Rerank？"]) == 0
    search_output = capsys.readouterr().out
    assert "source: fusion" in search_output
    assert "rag_notes.md" in search_output

    assert main([*common, "ask", "这个系统怎么做增量索引？"]) == 0
    ask_output = capsys.readouterr().out
    assert "[C1]" in ask_output
    assert "rag_design.md" in ask_output

    assert main([*common, "stats"]) == 0
    stats_output = capsys.readouterr().out
    assert "Chunk 数：" in stats_output
    assert "向量索引：已构建" in stats_output

    assert main([*common, "eval", str(dataset)]) == 0
    eval_output = capsys.readouterr().out
    for label in ["Hit@1", "Hit@3", "MRR", "引用覆盖率"]:
        assert label in eval_output

    cache_dir = tmp_path / "cache" / "embeddings"
    cached_files = {path.name for path in cache_dir.glob("*.json")}
    unrelated = tmp_path / "storage" / "keep.me"
    unrelated.write_text("keep", encoding="utf-8")

    assert main([*common, "rebuild", str(docs)]) == 0
    rebuild_output = capsys.readouterr().out
    assert "全量重建" in rebuild_output
    assert "Embedding 缓存命中：" in rebuild_output
    assert cached_files
    assert {path.name for path in cache_dir.glob("*.json")} == cached_files
    assert unrelated.read_text(encoding="utf-8") == "keep"
    assert main([*common, "search", "RRF 融合"]) == 0


def test_acceptance_second_index_is_unchanged(tmp_path, capsys):
    docs = tmp_path / "docs"
    shutil.copytree(PROJECT_ROOT / "examples" / "docs", docs)
    config = write_config(tmp_path / "config.yaml")
    command = ["--config", str(config), "index", str(docs)]

    assert main(command) == 0
    capsys.readouterr()
    assert main(command) == 0
    output = capsys.readouterr().out

    assert "新增文件：0 个" in output
    assert "更新文件：0 个" in output
    assert "跳过未变化文件：4 个" in output

