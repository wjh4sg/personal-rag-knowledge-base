from pathlib import Path

import yaml

from personal_rag.config import load_config


def write_config(root: Path) -> Path:
    path = root / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "chunk": {"size": 700, "overlap": 120},
                "retrieval": {
                    "top_k": 5,
                    "vector_top_k": 10,
                    "bm25_top_k": 10,
                },
                "provider": {
                    "mode": "mock",
                    "base_url": "https://api-inference.modelscope.cn/v1",
                    "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
                    "embedding_dimensions": 1024,
                    "llm_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                    "timeout_seconds": 90,
                    "fallback_to_mock": True,
                },
                "storage": {
                    "base_dir": "data/storage",
                    "cache_dir": "data/cache",
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_environment_overrides_modelscope_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("MODELSCOPE_API_KEY", "secret")

    config = load_config(write_config(tmp_path))

    assert config.provider.api_key == "secret"


def test_storage_paths_are_resolved_from_config_directory(tmp_path):
    config = load_config(write_config(tmp_path))

    assert config.storage.base_dir == tmp_path / "data/storage"
    assert config.storage.cache_dir == tmp_path / "data/cache"

