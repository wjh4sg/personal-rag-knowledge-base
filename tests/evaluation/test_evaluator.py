import json

import pytest

from personal_rag.core.schema import Answer, Citation, RetrievedChunk
from personal_rag.evaluation.evaluator import (
    citation_coverage,
    evaluate,
    hit_at_k,
    reciprocal_rank,
)


def result(source_path: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=source_path,
        text="text",
        score=1.0,
        source="fusion",
        metadata={"source_path": source_path},
    )


def test_reciprocal_rank_uses_first_expected_source():
    results = [result("other.md"), result("expected.md"), result("expected.md")]

    assert reciprocal_rank(results, ["expected.md"]) == pytest.approx(0.5)


def test_hit_at_k_only_checks_requested_prefix():
    results = [result("other.md"), result("expected.md")]

    assert hit_at_k(results, ["expected.md"], 1) == 0
    assert hit_at_k(results, ["expected.md"], 2) == 1


def test_citation_coverage_counts_answers_with_legal_citations():
    citation = Citation("C1", "chunk", "source.md", None, None)
    with_citation = Answer("q1", "a1", [citation], ["chunk"])
    without_citation = Answer("q2", "a2", [], [])

    assert citation_coverage([with_citation, without_citation]) == 0.5


class StaticRetriever:
    def retrieve(self, question, top_k=None):
        return [result("expected.md")]


class StaticPipeline:
    def ask(self, question):
        citation = Citation("C1", "expected.md", "expected.md", None, None)
        return Answer(question, "answer [C1]", [citation], ["expected.md"])


def test_evaluate_reports_retrieval_and_citation_metrics(tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "question": "question",
                    "expected_sources": ["expected.md"],
                    "ground_truth": "answer",
                }
            ]
        ),
        encoding="utf-8",
    )

    report = evaluate(dataset, StaticRetriever(), StaticPipeline())

    assert report.sample_count == 1
    assert report.hit_at_1 == 1.0
    assert report.hit_at_3 == 1.0
    assert report.mrr == 1.0
    assert report.citation_coverage == 1.0

