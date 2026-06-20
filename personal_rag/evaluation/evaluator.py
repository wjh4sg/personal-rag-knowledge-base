from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from personal_rag.core.schema import Answer, EvaluationReport, RetrievedChunk


def hit_at_k(
    results: list[RetrievedChunk],
    expected_sources: list[str],
    k: int,
) -> int:
    expected = set(expected_sources)
    return int(
        any(
            str(item.metadata.get("source_path", "")) in expected
            for item in results[:k]
        )
    )


def reciprocal_rank(
    results: list[RetrievedChunk],
    expected_sources: list[str],
) -> float:
    expected = set(expected_sources)
    for index, item in enumerate(results, start=1):
        if str(item.metadata.get("source_path", "")) in expected:
            return 1.0 / index
    return 0.0


def citation_coverage(answers: list[Answer]) -> float:
    if not answers:
        return 0.0
    return sum(bool(answer.citations) for answer in answers) / len(answers)


def evaluate(dataset_path: Path, retriever: Any, pipeline: Any) -> EvaluationReport:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(dataset, list):
        raise ValueError("Evaluation dataset must contain a JSON list")

    hit_1: list[int] = []
    hit_3: list[int] = []
    reciprocal_ranks: list[float] = []
    answers: list[Answer] = []
    for item in dataset:
        question = str(item["question"])
        expected = [str(source) for source in item["expected_sources"]]
        results = retriever.retrieve(question, top_k=5)
        hit_1.append(hit_at_k(results, expected, 1))
        hit_3.append(hit_at_k(results, expected, 3))
        reciprocal_ranks.append(reciprocal_rank(results, expected))
        answers.append(pipeline.ask(question))

    count = len(dataset)
    return EvaluationReport(
        sample_count=count,
        hit_at_1=sum(hit_1) / count if count else 0.0,
        hit_at_3=sum(hit_3) / count if count else 0.0,
        mrr=sum(reciprocal_ranks) / count if count else 0.0,
        citation_coverage=citation_coverage(answers),
        answers_without_citations=sum(not answer.citations for answer in answers),
    )

