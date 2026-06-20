from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from personal_rag.config import AppConfig, load_config
from personal_rag.evaluation.evaluator import evaluate
from personal_rag.ingest.indexer import Indexer
from personal_rag.providers import ProviderError
from personal_rag.providers.embeddings import APIEmbeddingClient, MockEmbeddingClient
from personal_rag.providers.generators import APIGenerator, MockGenerator
from personal_rag.rag.pipeline import RAGPipeline
from personal_rag.rag.retriever import HybridRetriever
from personal_rag.storage.bm25_store import BM25Store
from personal_rag.storage.chunk_store import ChunkStore
from personal_rag.storage.doc_store import DocStore
from personal_rag.storage.embedding_cache import EmbeddingCache
from personal_rag.storage.stats_store import StatsStore
from personal_rag.storage.vector_store import VectorStore


class MissingIndexError(RuntimeError):
    pass


@dataclass
class Services:
    config: AppConfig
    doc_store: DocStore
    chunk_store: ChunkStore
    vector_store: VectorStore
    bm25_store: BM25Store
    embedding_cache: EmbeddingCache
    stats_store: StatsStore
    embedding_client: Any


def _embedding_client(config: AppConfig) -> Any:
    if config.provider.mode == "api":
        return APIEmbeddingClient(
            base_url=config.provider.base_url,
            api_key=config.provider.api_key,
            model_name=config.provider.embedding_model,
            dimensions=config.provider.embedding_dimensions,
            timeout_seconds=config.provider.timeout_seconds,
        )
    if config.provider.mode == "mock":
        return MockEmbeddingClient(dimensions=config.provider.embedding_dimensions)
    raise ValueError(f"Unsupported provider mode: {config.provider.mode}")


def _generator(config: AppConfig) -> Any:
    if config.provider.mode == "api":
        return APIGenerator(
            base_url=config.provider.base_url,
            api_key=config.provider.api_key,
            model_name=config.provider.llm_model,
            timeout_seconds=config.provider.timeout_seconds,
        )
    return MockGenerator()


def _services(config_path: str | Path) -> Services:
    config = load_config(config_path)
    base = config.storage.base_dir
    return Services(
        config=config,
        doc_store=DocStore(base / "docs.json"),
        chunk_store=ChunkStore(base / "chunks.jsonl"),
        vector_store=VectorStore(base / "chroma", "personal_rag_chunks"),
        bm25_store=BM25Store(base / "bm25.pkl"),
        embedding_cache=EmbeddingCache(config.storage.cache_dir / "embeddings"),
        stats_store=StatsStore(base / "stats.json"),
        embedding_client=_embedding_client(config),
    )


def _require_index(services: Services) -> None:
    if (
        not services.chunk_store.exists()
        or not services.bm25_store.exists()
        or not services.vector_store.exists()
    ):
        raise MissingIndexError(
            "知识库索引不存在或不完整，请先运行 rag index <文档目录>，"
            "或运行 rag rebuild <文档目录> 全量重建。"
        )
    services.bm25_store.load()


def _retriever(services: Services) -> HybridRetriever:
    return HybridRetriever(
        embedding_client=services.embedding_client,
        vector_store=services.vector_store,
        bm25_store=services.bm25_store,
        vector_top_k=services.config.retrieval.vector_top_k,
        bm25_top_k=services.config.retrieval.bm25_top_k,
        top_k=services.config.retrieval.top_k,
    )


def _pipeline(services: Services, retriever: HybridRetriever) -> RAGPipeline:
    generator = _generator(services.config)
    return RAGPipeline(
        retriever,
        generator,
        fallback_generator=MockGenerator(),
        fallback_to_mock=(
            services.config.provider.mode == "api"
            and services.config.provider.fallback_to_mock
        ),
    )


def run_index(services: Services, docs_path: Path) -> int:
    report = Indexer(
        doc_store=services.doc_store,
        chunk_store=services.chunk_store,
        vector_store=services.vector_store,
        bm25_store=services.bm25_store,
        embedding_cache=services.embedding_cache,
        stats_store=services.stats_store,
        embedding_client=services.embedding_client,
        chunk_size=services.config.chunk.size,
        overlap=services.config.chunk.overlap,
    ).index(docs_path)
    print(f"扫描文件：{report.scanned} 个")
    print(f"新增文件：{report.added} 个")
    print(f"更新文件：{report.modified} 个")
    print(f"删除文件：{report.deleted} 个")
    print(f"跳过未变化文件：{report.unchanged} 个")
    print(f"忽略不支持文件：{report.unsupported_count} 个")
    print(f"逻辑文档数：{report.document_count}")
    print(f"总 Chunk 数：{report.chunk_count}")
    print(f"跳过 PDF 空页：{report.skipped_pdf_pages}")
    print(f"Embedding 缓存命中：{report.embedding_cache_hits}")
    print("Chroma 向量索引：完成")
    print("BM25 关键词索引：完成")
    return 0


def reset_index(services: Services) -> None:
    services.doc_store.reset()
    services.chunk_store.reset()
    services.bm25_store.reset()
    services.stats_store.reset()
    services.vector_store.reset()


def run_rebuild(services: Services, docs_path: Path) -> int:
    if not docs_path.is_dir():
        raise FileNotFoundError(f"Document directory does not exist: {docs_path}")
    print("清理现有索引并开始全量重建（保留 Embedding 缓存）……")
    reset_index(services)
    return run_index(services, docs_path)


def run_search(services: Services, query: str) -> int:
    _require_index(services)
    results = _retriever(services).retrieve(query)
    print(f"Top {len(results)} 检索结果：")
    for index, item in enumerate(results, start=1):
        source_path = item.metadata.get("source_path", "")
        page = item.metadata.get("page") or ""
        heading = item.metadata.get("heading") or ""
        location = f"{source_path}"
        if page:
            location += f" 第 {page} 页"
        if heading:
            location += f"#{heading}"
        preview = " ".join(item.text.split())[:240]
        print(f"\n[{index}] {location}")
        print(f"score: {item.score:.6f}")
        print(f"source: {item.source}")
        print(f"内容：{preview}")
    return 0


def run_ask(services: Services, question: str) -> int:
    _require_index(services)
    retriever = _retriever(services)
    answer = _pipeline(services, retriever).ask(question)
    print("回答：")
    print(answer.answer)
    print(f"\n生成模式：{answer.generation_mode}")
    print("\n引用：")
    if not answer.citations:
        print("无合法引用")
    for citation in answer.citations:
        location = citation.source_path
        if citation.page:
            location += f" 第 {citation.page} 页"
        if citation.heading:
            location += f"#{citation.heading}"
        print(f"[{citation.citation_id}] {location}")
    return 0


def run_stats(services: Services) -> int:
    documents = services.doc_store.load()
    chunks = services.chunk_store.load_all()
    stats = services.stats_store.load()
    logical_documents = sum(len(info.get("doc_ids", [])) for info in documents.values())
    print(f"文件数：{len(documents)}")
    print(f"文档数：{logical_documents}")
    print(f"Chunk 数：{len(chunks)}")
    print(f"向量索引：{'已构建' if services.vector_store.exists() else '未构建'}")
    print(f"BM25 索引：{'已构建' if services.bm25_store.exists() else '未构建'}")
    print(f"Embedding 模式：{stats.get('embedding_mode', services.config.provider.mode)}")
    print(
        "Embedding 模型："
        f"{stats.get('embedding_model', services.embedding_client.model_name)}"
    )
    print(f"最近索引时间：{stats.get('last_indexed_at', '从未')}")
    return 0


def run_eval(services: Services, dataset_path: Path) -> int:
    _require_index(services)
    retriever = _retriever(services)
    report = evaluate(dataset_path, retriever, _pipeline(services, retriever))
    print(f"评估样本数：{report.sample_count}")
    print(f"Hit@1: {report.hit_at_1:.2f}")
    print(f"Hit@3: {report.hit_at_3:.2f}")
    print(f"MRR: {report.mrr:.2f}")
    print(f"引用覆盖率：{report.citation_coverage:.2f}")
    print(f"无引用回答数：{report.answers_without_citations}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="建立或增量更新知识库索引")
    index_parser.add_argument("docs_path", type=Path)

    rebuild_parser = subparsers.add_parser(
        "rebuild",
        help="保留 Embedding 缓存并全量重建知识库索引",
    )
    rebuild_parser.add_argument("docs_path", type=Path)

    search_parser = subparsers.add_parser("search", help="查看混合检索结果")
    search_parser.add_argument("query")

    ask_parser = subparsers.add_parser("ask", help="生成带引用的知识库答案")
    ask_parser.add_argument("question")

    subparsers.add_parser("stats", help="查看知识库状态")

    eval_parser = subparsers.add_parser("eval", help="运行检索与引用评估")
    eval_parser.add_argument("dataset_path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        return int(error.code)

    services: Services | None = None
    try:
        services = _services(args.config)
        if args.command == "index":
            return run_index(services, args.docs_path)
        if args.command == "rebuild":
            return run_rebuild(services, args.docs_path)
        if args.command == "search":
            return run_search(services, args.query)
        if args.command == "ask":
            return run_ask(services, args.question)
        if args.command == "stats":
            return run_stats(services)
        if args.command == "eval":
            return run_eval(services, args.dataset_path)
        parser.print_usage(sys.stderr)
        return 2
    except MissingIndexError as error:
        print(f"错误：{error}", file=sys.stderr)
        return 3
    except ProviderError as error:
        print(f"模型服务错误：{error}", file=sys.stderr)
        return 4
    except (FileNotFoundError, ValueError) as error:
        print(f"输入错误：{error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"运行失败：{error}", file=sys.stderr)
        return 1
    finally:
        if services is not None:
            services.vector_store.close()


if __name__ == "__main__":
    raise SystemExit(main())
