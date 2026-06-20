import pytest

from personal_rag.providers.generators import APIGenerator, MockGenerator, ProviderError


class FakeTransport:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.last_json = None

    def __call__(self, url, headers, payload, timeout):
        self.last_json = payload
        if self.error:
            raise self.error
        return self.response


def make_generator(transport):
    return APIGenerator(
        base_url="https://api.example/v1",
        api_key="secret-token",
        model_name="chat-model",
        timeout_seconds=5,
        transport=transport,
    )


def test_modelscope_chat_returns_content():
    generator = make_generator(
        FakeTransport({"choices": [{"message": {"content": "回答。[C1]"}}]})
    )

    assert generator.generate("prompt") == "回答。[C1]"


def test_modelscope_chat_rejects_empty_choices():
    generator = make_generator(FakeTransport({"choices": None}))

    with pytest.raises(ProviderError, match="empty"):
        generator.generate("prompt")


def test_provider_error_never_includes_api_key():
    generator = make_generator(FakeTransport(error=RuntimeError("failed secret-token")))

    with pytest.raises(ProviderError) as captured:
        generator.generate("prompt")

    assert "secret-token" not in str(captured.value)


def test_mock_generator_uses_first_context_citation():
    prompt = "可用资料：\n\n[C1] 来源：notes.md\n系统使用内容 Hash 判断变化。"

    answer = MockGenerator().generate(prompt)

    assert "内容 Hash" in answer
    assert "[C1]" in answer

