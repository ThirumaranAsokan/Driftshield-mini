from __future__ import annotations

import pytest

from benchmarks.convert_fintrace_review import convert_records


def _record(review_id: str, labels: dict, split: str = "calibration") -> dict:
    return {
        "review_id": review_id,
        "split": split,
        "task_goal": "Find the latest revenue.",
        "trajectory_excerpt": [
            {"turn": 1, "type": "message", "role": "user", "content": "Find revenue."},
            {"turn": 1, "type": "message", "role": "assistant", "content": "I will search the filing."},
            {"turn": 1, "type": "function_call", "tool_name": "search_reports", "arguments": "{\"query\":\"revenue\"}"},
            {"turn": 1, "type": "function_call_output", "tool_name": "search_reports", "content": "Found a filing."},
        ],
        "labels": labels,
        "reviewer_id": "reviewer-a",
        "review_confidence": "high",
    }


def _resolved(action_loop="negative", goal_drift="negative", resource_spike="negative"):
    return {
        "action_loop": action_loop,
        "goal_drift": goal_drift,
        "resource_spike": resource_spike,
    }


def test_converter_maps_labels_events_and_run_specific_goal():
    payload = {"runs": [_record("abc123", _resolved(goal_drift="positive"))]}

    result = convert_records(payload, "calibration")
    scenario = result["scenarios"][0]
    run = scenario["runs"][0]

    assert result["included_runs"] == 1
    assert len(result["scenarios"]) == 1
    assert scenario["name"] == "fintrace-review"
    assert run["run_id"] == "abc123"
    assert run["expected_detectors"] == ["goal_drift"]
    assert run["metadata"]["labels_status"] == "labelled"
    assert [event["action_type"] for event in run["events"]] == [
        "llm_request", "tool_call", "tool_result"
    ]
    assert run["events"][0]["output_data"]["text"] == "I will search the filing."
    assert run["events"][0]["input_data"]["input"] == "Find the latest revenue."
    assert run["events"][0]["metadata"]["goal_description"] == "Find the latest revenue."
    assert run["events"][1]["action_name"] == "search_reports"
    assert scenario["calibration_runs"] == 1


def test_uncertain_and_blank_labels_are_excluded_not_negative():
    payload = {"runs": [
        _record("resolved", _resolved()),
        _record("uncertain", _resolved(action_loop="uncertain", resource_spike="")),
    ]}

    result = convert_records(payload, "calibration")
    scenario = result["scenarios"][0]

    assert result["included_runs"] == 1
    assert result["excluded_unresolved_count"] == 1
    assert result["excluded_unresolved"][0]["review_id"] == "uncertain"
    assert scenario["runs"][0]["expected_detectors"] == []


def test_converter_rejects_unsupported_labels_and_empty_split():
    with pytest.raises(ValueError, match="Unsupported review label"):
        convert_records({"runs": [_record("bad", _resolved(action_loop="maybe"))]}, "calibration")

    with pytest.raises(ValueError, match="No fully labelled records"):
        convert_records({"runs": [_record("blank", _resolved(action_loop="", goal_drift="uncertain"))]}, "holdout")


def test_converter_only_uses_requested_split_and_keeps_one_scenario():
    payload = {"runs": [
        _record("cal", _resolved(), "calibration"),
        _record("hold", _resolved(action_loop="positive"), "holdout"),
    ]}

    result = convert_records(payload, "holdout")

    assert result["included_runs"] == 1
    assert len(result["scenarios"]) == 1
    assert result["scenarios"][0]["runs"][0]["run_id"] == "hold"
    assert result["scenarios"][0]["calibration_runs"] == 2
    assert "does not transfer a frozen calibration baseline" in result["evaluation_note"]


def test_duplicate_review_ids_in_requested_split_are_rejected():
    payload = {"runs": [
        _record("duplicate", _resolved()),
        _record("duplicate", _resolved()),
    ]}
    with pytest.raises(ValueError, match="Duplicate review_id"):
        convert_records(payload, "calibration")
