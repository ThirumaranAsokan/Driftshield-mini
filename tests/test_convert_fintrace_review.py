from __future__ import annotations

import pytest

from benchmarks.convert_fintrace_review import convert_records


def _record(review_id: str, labels: dict, split: str = "calibration") -> dict:
    return {
        "review_id": review_id,
        "split": split,
        "task_goal": "Find the latest revenue.",
        "trajectory_excerpt": [
            {"turn": 1, "type": "message", "role": "assistant", "content": "I will search the filing."},
            {"turn": 1, "type": "function_call", "tool_name": "search_reports", "arguments": "{\"query\":\"revenue\"}"},
            {"turn": 1, "type": "function_call_output", "tool_name": "search_reports", "content": "Found a filing."},
        ],
        "labels": labels,
        "reviewer_id": "reviewer-a",
        "review_confidence": "high",
    }


def test_converter_maps_resolved_labels_and_keeps_events_in_order():
    payload = {"runs": [_record("abc123", {
        "action_loop": "negative",
        "goal_drift": "positive",
        "resource_spike": False,
    })]}

    result = convert_records(payload, "calibration")
    run = result["scenarios"][0]["runs"][0]

    assert result["included_runs"] == 1
    assert run["run_id"] == "abc123"
    assert run["expected_detectors"] == ["goal_drift"]
    assert run["metadata"]["labels_status"] == "labelled"
    assert [event["action_type"] for event in run["events"]] == [
        "llm_request", "tool_call", "tool_result"
    ]
    assert run["events"][0]["output_data"]["text"] == "I will search the filing."
    assert run["events"][1]["action_name"] == "search_reports"


def test_uncertain_and_blank_labels_are_excluded_not_negative():
    payload = {"runs": [
        _record("resolved", {"action_loop": "negative", "goal_drift": "negative", "resource_spike": "negative"}),
        _record("uncertain", {"action_loop": "uncertain", "goal_drift": "negative", "resource_spike": ""}),
    ]}

    result = convert_records(payload, "calibration")

    assert result["included_runs"] == 1
    assert result["excluded_unresolved_count"] == 1
    assert result["excluded_unresolved"][0]["review_id"] == "uncertain"
    assert result["scenarios"][0]["runs"][0]["expected_detectors"] == []


def test_converter_rejects_unsupported_labels_and_empty_split():
    with pytest.raises(ValueError, match="Unsupported review label"):
        convert_records({"runs": [_record("bad", {
            "action_loop": "maybe",
            "goal_drift": "negative",
            "resource_spike": "negative",
        })]}, "calibration")

    with pytest.raises(ValueError, match="No fully labelled records"):
        convert_records({"runs": [_record("blank", {
            "action_loop": "",
            "goal_drift": "uncertain",
            "resource_spike": "negative",
        })]}, "holdout")


def test_converter_only_uses_requested_split():
    payload = {"runs": [
        _record("cal", {"action_loop": False, "goal_drift": False, "resource_spike": False}, "calibration"),
        _record("hold", {"action_loop": True, "goal_drift": False, "resource_spike": False}, "holdout"),
    ]}

    result = convert_records(payload, "holdout")

    assert result["included_runs"] == 1
    assert result["scenarios"][0]["runs"][0]["run_id"] == "hold"
