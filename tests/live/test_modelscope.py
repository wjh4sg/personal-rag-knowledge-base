import os

import pytest

from personal_rag.providers.embeddings import APIEmbeddingClient
from personal_rag.providers.generators import APIGenerator


@pytest.mark.live
@pytest.mark.skipif(
    not os.getenv("MODELSCOPE_API_KEY"),
    reason="MODELSCOPE_API_KEY is not configured",
)
def test_modelscope_embedding_and_chat():
    api_key = os.environ["MODELSCOPE_API_KEY"]
    embedding = APIEmbeddingClient(
        base_url="https://api-inference.modelscope.cn/v1",
        api_key=api_key,
        model_name="Qwen/Qwen3-Embedding-0.6B",
        dimensions=1024,
        timeout_seconds=90,
    )
    generator = APIGenerator(
        base_url="https://api-inference.modelscope.cn/v1",
        api_key=api_key,
        model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        timeout_seconds=90,
    )

    assert len(embedding.embed(["RAG 检索连接测试"])[0]) == 1024
    assert generator.generate("只回答：连接成功").strip()

