"""Tests for the blinded FinTrace review-set builder."""

from __future__ import annotations

from benchmarks.build_fintrace_review_set import (
    build_review_records,
    stable_source_id,
    trajectory_excerpt,
)


def _row(identifier: int, text: str = "Find revenue") -> dict:
    return {
        "id": identifier,
        "task_type": "financial_qa",
        "source_query": text,
        "output_answer": "must not appear",
        "reference_answer": "must not appear",
        "golden_trajectories": [{"secret": "must not appear"}],
        "output_trajectory": [
            {
                "turn": 1,
                "output": [
                    {"type": "message", "role": "assistant", "content": "Searching report"},
                    {
                        "type": "function_call",
                        "name": "search_reports",
                        "arguments": "{\"query\": \"revenue\"}",
                    },
                    {"type": "function_call_output", "name": "search_reports", "output": "Found result"},
                ],
            }
        ],
    }


def test_review_set_is_deterministic_and_split_without_predictions() -> None:
    rows = [_row(index) for index in range(10)]
    first = build_review_records(rows, limit=8, seed=17, calibration_count=5)
    second = build_review_records(rows, limit=8, seed=17, calibration_count=5)

    assert first == second
    assert len(first) == 8
    assert sum(item["split"] == "calibration" for item in first) == 5
    assert sum(item["split"] == "holdout" for item in first) == 3
    assert all(
        item["labels"] == {"action_loop": "", "goal_drift": "", "resource_spike": ""}
        for item in first
    )
    assert all(
        item["review_reason"] == {"action_loop": "", "goal_drift": "", "resource_spike": ""}
        for item in first
    )
    assert all("output_answer" not in item and "reference_answer" not in item for item in first)
    assert all("golden_trajectories" not in item for item in first)
    assert all("must not appear" not in str(item) for item in first)


def test_trajectory_excerpt_keeps_actions_and_observations() -> None:
    excerpt = trajectory_excerpt(_row(1)["output_trajectory"])
    assert [item["type"] for item in excerpt] == ["message", "function_call", "function_call_output"]
    assert excerpt[1]["tool_name"] == "search_reports"
    assert excerpt[1]["arguments"] == '{"query": "revenue"}'


def test_stable_source_id_is_repeatable() -> None:
    row = _row(42)
    assert stable_source_id(row) == stable_source_id(row)
    assert len(stable_source_id(row)) == 16


def test_empty_dataset_and_invalid_split_size() -> None:
    assert build_review_records([], limit=3, seed=1, calibration_count=2) == []
    try:
        build_review_records([_row(1)], limit=1, calibration_count=2)
    except ValueError as exc:
        assert "calibration_count" in str(exc)
    else:
        raise AssertionError("expected invalid calibration_count to fail")
