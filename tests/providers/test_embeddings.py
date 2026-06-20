import math

import pytest

from personal_rag.providers.embeddings import APIEmbeddingClient, MockEmbeddingClient


class FakeTransport:
    def __init__(self, response):
        self.response = response
        self.last_url = None
        self.last_headers = None
        self.last_json = None

    def __call__(self, url, headers, payload, timeout):
        self.last_url = url
        self.last_headers = headers
        self.last_json = payload
        return self.response


def test_mock_embeddings_are_deterministic_and_normalized():
    client = MockEmbeddingClient(dimensions=32)

    first = client.embed(["混合检索"])[0]
    second = client.embed(["混合检索"])[0]

    assert first == second
    assert math.sqrt(sum(value * value for value in first)) == pytest.approx(1.0)


def test_mock_embeddings_return_zero_vector_for_empty_text():
    assert MockEmbeddingClient(dimensions=4).embed([""])[0] == [0.0, 0.0, 0.0, 0.0]


def test_modelscope_embedding_accepts_nested_response():
    transport = FakeTransport({"data": {"data": [{"embedding": [0.1, 0.2]}]}})
    client = APIEmbeddingClient(
        base_url="https://api.example/v1",
        api_key="secret",
        model_name="embedding-model",
        dimensions=2,
        timeout_seconds=5,
        transport=transport,
    )

    assert client.embed(["text"]) == [[0.1, 0.2]]
    assert transport.last_url == "https://api.example/v1/embeddings"
    assert transport.last_json["encoding_format"] == "float"


def test_modelscope_embedding_accepts_standard_response():
    transport = FakeTransport({"data": [{"embedding": [0.1, 0.2]}]})
    client = APIEmbeddingClient(
        base_url="https://api.example/v1",
        api_key="secret",
        model_name="embedding-model",
        dimensions=2,
        transport=transport,
    )

    assert client.embed(["text"]) == [[0.1, 0.2]]


def test_modelscope_embedding_rejects_wrong_dimensions():
    client = APIEmbeddingClient(
        base_url="https://api.example/v1",
        api_key="secret",
        model_name="embedding-model",
        dimensions=3,
        transport=FakeTransport({"data": [{"embedding": [0.1, 0.2]}]}),
    )

    with pytest.raises(ValueError, match="dimensions"):
        client.embed(["text"])

