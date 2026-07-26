"""Offline unit tests for the consolidated evaluation helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tests" / "evaluate_retrieval.py"
SPEC = importlib.util.spec_from_file_location("evaluate_retrieval", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def test_default_golden_path_exists_and_has_required_question_count() -> None:
    rows = evaluation.load_golden(ROOT / "golden_dataset.jsonl")
    assert len(rows) == 14
    assert all(evaluation.REQUIRED_GOLD_FIELDS <= set(row) for row in rows)


def test_deterministic_rank_metrics_for_valid_and_missed_results() -> None:
    valid = evaluation.deterministic_rank_metrics(
        ["wrong", "target", "other"], {"target"}, top_k=3
    )
    missed = evaluation.deterministic_rank_metrics(
        ["wrong", "other"], {"target"}, top_k=2
    )

    assert valid["hit_at_k"] == 1.0
    assert valid["precision_at_k"] == 1 / 3
    assert valid["recall_at_k"] == 1.0
    assert valid["mrr_at_k"] == 0.5
    assert 0 < valid["ndcg_at_k"] < 1
    assert all(value == 0.0 for value in missed.values())


def test_ordered_unique_prevents_duplicate_document_credit() -> None:
    assert evaluation.ordered_unique(["doc-a", "doc-a", "doc-b"]) == [
        "doc-a",
        "doc-b",
    ]


def test_score_value_accepts_ragas_result_shape() -> None:
    result_type = type("MetricResult", (), {"value": 0.75})
    assert evaluation._score_value(result_type()) == 0.75


def test_pinned_ragas_exposes_all_four_required_metric_classes() -> None:
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    assert all(
        callable(metric)
        for metric in (AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness)
    )
