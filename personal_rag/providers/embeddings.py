from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from typing import Any

import jieba
import requests

from personal_rag.providers import ProviderError

Transport = Callable[[str, dict[str, str], dict[str, Any], int], dict[str, Any]]


def _requests_transport(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise ValueError("Provider response must be a JSON object")
    return value


class MockEmbeddingClient:
    provider_name = "mock"
    model_name = "mock-hash-embedding"

    def __init__(self, dimensions: int = 128):
        if dimensions <= 0:
            raise ValueError("Embedding dimensions must be positive")
        self.dimensions = dimensions

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = [token.strip().lower() for token in jieba.lcut(text) if token.strip()]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimensions
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]


class APIEmbeddingClient:
    provider_name = "api"

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model_name: str,
        dimensions: int,
        timeout_seconds: int = 90,
        transport: Transport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.dimensions = dimensions
        self.timeout_seconds = timeout_seconds
        self.transport = transport or _requests_transport

    def _extract_vectors(self, response: dict[str, Any]) -> list[list[float]]:
        data = response.get("data")
        if isinstance(data, dict):
            data = data.get("data")
        if not isinstance(data, list):
            raise ProviderError("Embedding provider returned an empty or invalid response")
        vectors: list[list[float]] = []
        for item in data:
            if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
                raise ProviderError("Embedding provider returned an invalid vector")
            vector = [float(value) for value in item["embedding"]]
            if len(vector) != self.dimensions:
                raise ValueError(
                    f"Embedding dimensions mismatch: expected {self.dimensions}, got {len(vector)}"
                )
            vectors.append(vector)
        return vectors

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise ProviderError("MODELSCOPE_API_KEY is required in API mode")
        payload = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float",
        }
        try:
            response = self.transport(
                f"{self.base_url}/embeddings",
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                payload,
                self.timeout_seconds,
            )
            return self._extract_vectors(response)
        except (ProviderError, ValueError):
            raise
        except Exception as error:
            message = str(error).replace(self.api_key, "[REDACTED]")
            raise ProviderError(f"Embedding provider request failed: {message}") from None

