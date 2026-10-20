import json

from benchmarks.validate_trace_dataset import validate_dataset


def _write_dataset(tmp_path, labels_status, events=None, expected_detectors=None):
    run = {
        "run_id": "run-001",
        "expected_detectors": expected_detectors or [],
        "events": events or [],
        "metadata": {"labels_status": labels_status},
    }
    payload = {
        "scenarios": [
            {
                "name": "finance-agent",
                "goal": "Summarise a financial report",
                "calibration_runs": 1,
                "runs": [run],
            }
        ]
    }
    path = tmp_path / "traces.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _repeated_tool_events():
    return [
        {"action_type": "tool_call", "action_name": "search"}
        for _ in range(4)
    ]


def test_unlabelled_runs_are_excluded_even_when_a_detector_fires(tmp_path):
    report = validate_dataset(
        _write_dataset(tmp_path, "unlabelled", events=_repeated_tool_events())
    )

    assert report["labelled_runs"] == 0
    assert report["unlabelled_runs"] == 1
    assert report["runs_detail"][0]["labels_status"] == "unlabelled"
    assert "action_loop" in report["runs_detail"][0]["predicted"]
    for metric in report["metrics"].values():
        assert (metric["tp"], metric["fp"], metric["fn"], metric["tn"]) == (0, 0, 0, 0)
        assert metric["precision"] is None
        assert metric["recall"] is None
        assert metric["f1"] is None
        assert metric["false_positive_rate"] is None


def test_labelled_negative_run_does_not_claim_recall_without_positive_examples(tmp_path):
    report = validate_dataset(_write_dataset(tmp_path, "labelled"))

    assert report["labelled_runs"] == 1
    assert report["unlabelled_runs"] == 0
    for metric in report["metrics"].values():
        assert metric["tn"] == 1
        assert metric["tp"] == metric["fp"] == metric["fn"] == 0
        assert metric["precision"] is None
        assert metric["recall"] is None
        assert metric["f1"] is None
        assert metric["false_positive_rate"] == 0.0


def test_labelled_false_positive_has_precision_and_fpr_but_no_recall(tmp_path):
    report = validate_dataset(
        _write_dataset(tmp_path, "labelled", events=_repeated_tool_events())
    )

    metric = report["metrics"]["action_loop"]
    assert metric["tp"] == 0
    assert metric["fp"] == 1
    assert metric["fn"] == 0
    assert metric["tn"] == 0
    assert metric["precision"] == 0.0
    assert metric["recall"] is None
    assert metric["f1"] is None
    assert metric["false_positive_rate"] == 1.0


def test_unknown_label_status_is_rejected(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="Unsupported labels_status"):
        validate_dataset(_write_dataset(tmp_path, "pending"))

def test_unlabelled_run_cannot_supply_expected_detector_labels(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="must not provide expected_detectors"):
        validate_dataset(
            _write_dataset(
                tmp_path,
                "unlabelled",
                expected_detectors=["action_loop"],
            )
        )
