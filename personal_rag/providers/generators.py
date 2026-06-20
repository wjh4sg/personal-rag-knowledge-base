from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

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


class MockGenerator:
    mode = "mock"

    def generate(self, prompt: str) -> str:
        match = re.search(
            r"\[C1\]\s+来源：[^\n]*\n(?P<text>.*?)(?=\n\n\[C\d+\]|\n\n请输出：|\Z)",
            prompt,
            flags=re.DOTALL,
        )
        evidence = match.group("text").strip() if match else ""
        if not evidence:
            return "当前知识库信息不足，无法确定。"
        preview = " ".join(evidence.split())
        return f"回答：\n根据检索资料，{preview}[C1]\n\n引用：\n[C1]"


class APIGenerator:
    mode = "api"

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model_name: str,
        timeout_seconds: int = 90,
        transport: Transport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.transport = transport or _requests_transport

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise ProviderError("MODELSCOPE_API_KEY is required in API mode")
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "你是严谨的 RAG 问答助手，只能根据提供的上下文回答并标注引用。",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "stream": False,
        }
        try:
            response = self.transport(
                f"{self.base_url}/chat/completions",
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                payload,
                self.timeout_seconds,
            )
            choices = response.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ProviderError("Chat provider returned an empty response")
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str) or not content.strip():
                raise ProviderError("Chat provider returned empty content")
            return content.strip()
        except ProviderError:
            raise
        except Exception as error:
            message = str(error).replace(self.api_key, "[REDACTED]")
            raise ProviderError(f"Chat provider request failed: {message}") from None


__all__ = ["APIGenerator", "MockGenerator", "ProviderError"]

