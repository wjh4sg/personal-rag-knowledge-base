from __future__ import annotations

from typing import Any

from personal_rag.core.schema import Answer
from personal_rag.providers import ProviderError
from personal_rag.rag.citations import check_citations
from personal_rag.rag.context import build_context

INSUFFICIENT_ANSWER = "当前知识库信息不足，无法确定。"


class RAGPipeline:
    def __init__(
        self,
        retriever: Any,
        generator: Any,
        *,
        fallback_generator: Any | None = None,
        fallback_to_mock: bool = False,
    ):
        self.retriever = retriever
        self.generator = generator
        self.fallback_generator = fallback_generator
        self.fallback_to_mock = fallback_to_mock

    def ask(self, question: str) -> Answer:
        retrieved_chunks = self.retriever.retrieve(question)
        if not retrieved_chunks:
            return Answer(
                question=question,
                answer=INSUFFICIENT_ANSWER,
                citations=[],
                used_chunks=[],
                generation_mode="none",
            )

        built = build_context(question, retrieved_chunks)
        generation_mode = getattr(self.generator, "mode", "unknown")
        try:
            answer_text = self.generator.generate(built.prompt)
        except ProviderError:
            if not self.fallback_to_mock or self.fallback_generator is None:
                raise
            answer_text = self.fallback_generator.generate(built.prompt)
            generation_mode = "mock-fallback"

        return Answer(
            question=question,
            answer=answer_text,
            citations=check_citations(answer_text, built.citation_map),
            used_chunks=[chunk.chunk_id for chunk in retrieved_chunks],
            generation_mode=generation_mode,
        )

