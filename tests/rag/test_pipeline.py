from personal_rag.core.schema import RetrievedChunk
from personal_rag.providers import ProviderError
from personal_rag.rag.pipeline import RAGPipeline
from personal_rag.rag.retriever import HybridRetriever


class EmptyRetriever:
    def retrieve(self, question, top_k=None):
        return []


class StaticRetriever:
    def retrieve(self, question, top_k=None):
        return [
            RetrievedChunk(
                chunk_id="chunk_1",
                text="内容 Hash 用于识别文件变化。",
                score=0.03,
                source="fusion",
                metadata={
                    "source_path": "design.md",
                    "page": None,
                    "heading": "增量索引",
                },
            )
        ]


class StaticGenerator:
    mode = "api"

    def generate(self, prompt):
        return "系统比较内容 Hash。[C1]"


class FailingGenerator:
    mode = "api"

    def generate(self, prompt):
        raise ProviderError("temporary failure")


class FallbackGenerator:
    mode = "mock"

    def generate(self, prompt):
        return "降级回答。[C1]"


def test_pipeline_returns_insufficient_without_results():
    answer = RAGPipeline(EmptyRetriever(), StaticGenerator()).ask("unknown")

    assert answer.answer == "当前知识库信息不足，无法确定。"
    assert answer.citations == []
    assert answer.used_chunks == []


def test_pipeline_returns_only_legal_citations():
    answer = RAGPipeline(StaticRetriever(), StaticGenerator()).ask("怎么索引？")

    assert answer.answer == "系统比较内容 Hash。[C1]"
    assert [citation.citation_id for citation in answer.citations] == ["C1"]
    assert answer.used_chunks == ["chunk_1"]
    assert answer.generation_mode == "api"


def test_pipeline_can_explicitly_fallback_to_mock():
    pipeline = RAGPipeline(
        StaticRetriever(),
        FailingGenerator(),
        fallback_generator=FallbackGenerator(),
        fallback_to_mock=True,
    )

    answer = pipeline.ask("怎么索引？")

    assert answer.generation_mode == "mock-fallback"
    assert answer.answer == "降级回答。[C1]"


class FakeEmbedding:
    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(texts)
        return [[1.0, 0.0]]


class FakeVectorStore:
    def query(self, embedding, top_k):
        return StaticRetriever().retrieve("")[0:top_k]


class FakeBM25Store:
    def search(self, query, top_k):
        return []


def test_hybrid_retriever_embeds_query_and_fuses_results():
    embedding = FakeEmbedding()
    retriever = HybridRetriever(
        embedding_client=embedding,
        vector_store=FakeVectorStore(),
        bm25_store=FakeBM25Store(),
        vector_top_k=10,
        bm25_top_k=10,
        top_k=5,
    )

    results = retriever.retrieve("增量索引")

    assert embedding.calls == [["增量索引"]]
    assert results[0].source == "fusion"

