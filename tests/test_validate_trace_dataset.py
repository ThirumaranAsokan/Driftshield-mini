import json

from benchmarks.validate_trace_dataset import validate_dataset


def _write_dataset(tmp_path, labels_status):
    payload = {
        "scenarios": [
            {
                "name": "finance-agent",
                "goal": "Summarise a financial report",
                "calibration_runs": 1,
                "runs": [
                    {
                        "run_id": "run-001",
                        "expected_detectors": [],
                        "events": [],
                        "metadata": {"labels_status": labels_status},
                    }
                ],
            }
        ]
    }
    path = tmp_path / "traces.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_unlabelled_runs_are_excluded_from_confusion_metrics(tmp_path):
    report = validate_dataset(_write_dataset(tmp_path, "unlabelled"))

    assert report["labelled_runs"] == 0
    assert report["unlabelled_runs"] == 1
    assert report["runs_detail"][0]["labels_status"] == "unlabelled"
    for metric in report["metrics"].values():
        assert (metric["tp"], metric["fp"], metric["fn"], metric["tn"]) == (0, 0, 0, 0)
        assert metric["precision"] is None
        assert metric["recall"] is None
        assert metric["f1"] is None
        assert metric["false_positive_rate"] is None


def test_labelled_negative_runs_still_count_as_true_negatives(tmp_path):
    report = validate_dataset(_write_dataset(tmp_path, "labelled"))

    assert report["labelled_runs"] == 1
    assert report["unlabelled_runs"] == 0
    for metric in report["metrics"].values():
        assert metric["tn"] == 1
        assert metric["tp"] == metric["fp"] == metric["fn"] == 0
        assert metric["precision"] == 0.0
        assert metric["recall"] == 0.0
        assert metric["f1"] == 0.0
        assert metric["false_positive_rate"] == 0.0


def test_unknown_label_status_is_rejected(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="Unsupported labels_status"):
        validate_dataset(_write_dataset(tmp_path, "pending"))
