"""Validate labelled real-agent traces against DriftShield detectors.

Input format:
{
  "scenarios": [
    {
      "name": "finance-agent",
      "goal": "Summarise financial reports",
      "calibration_runs": 3,
      "runs": [
        {
          "run_id": "run-001",
          "expected_detectors": [],
          "events": [
            {"action_type": "tool_call", "action_name": "search", "token_count": 120,
             "duration_ms": 25, "input_data": {}, "output_data": {}, "metadata": {}}
          ]
        }
      ]
    }
  ]
}

The harness does not invent labels. Expected detector sets must be supplied by
the dataset author. Runs explicitly marked unlabelled are excluded from confusion
matrices and accuracy metrics. Results are engineering measurements, not
production accuracy claims.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

from driftshield_mini import DriftMonitor
from driftshield_mini.models import DetectorType


def _detectors(values: list[str]) -> set[DetectorType]:
    return {DetectorType(value) for value in values}


def validate_dataset(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Dataset must contain a non-empty 'scenarios' list")

    counts = Counter()
    latency_samples: dict[str, list[int]] = {detector.value: [] for detector in DetectorType}
    run_results = []
    labelled_runs = 0
    unlabelled_runs = 0

    with tempfile.TemporaryDirectory(prefix="driftshield-validation-") as tmp:
        for scenario_index, scenario in enumerate(scenarios):
            name = str(scenario["name"])
            if not name.strip():
                raise ValueError("Scenario name must be non-empty")
            monitor = DriftMonitor(
                agent_id=name,
                db_path=str(Path(tmp) / f"scenario-{scenario_index}.db"),
                goal_description=str(scenario.get("goal", "")),
                calibration_runs=int(scenario.get("calibration_runs", 30)),
                loop_max_repeats=int(scenario.get("loop_max_repeats", 4)),
            )

            for run in scenario["runs"]:
                run_id = monitor.start_run(run_id=str(run["run_id"]))
                predicted: set[DetectorType] = set()
                first_detection_index: dict[DetectorType, int] = {}

                for event_index, event in enumerate(run["events"]):
                    detected = monitor.record_event(
                        str(event["action_type"]),
                        str(event["action_name"]),
                        run_id=run_id,
                        token_count=int(event.get("token_count", 0)),
                        duration_ms=float(event.get("duration_ms", 0.0)),
                        input_data=dict(event.get("input_data", {})),
                        output_data=dict(event.get("output_data", {})),
                        metadata=dict(event.get("metadata", {})),
                    )
                    for item in detected:
                        predicted.add(item.detector)
                        first_detection_index.setdefault(item.detector, event_index)

                metadata = run.get("metadata") or {}
                labels_status = str(metadata.get("labels_status", "labelled")).lower()
                if labels_status not in {"labelled", "unlabelled"}:
                    raise ValueError(
                        f"Unsupported labels_status {labels_status!r} for run {run_id!r}; "
                        "expected 'labelled' or 'unlabelled'"
                    )

                expected = _detectors(list(run.get("expected_detectors", [])))
                if labels_status == "unlabelled" and expected:
                    raise ValueError(
                        f"Unlabelled run {run_id!r} must not provide expected_detectors"
                    )
                if labels_status == "labelled":
                    labelled_runs += 1
                    for detector in DetectorType:
                        expected_hit = detector in expected
                        predicted_hit = detector in predicted
                        if expected_hit and predicted_hit:
                            counts[f"{detector.value}.tp"] += 1
                        elif not expected_hit and predicted_hit:
                            counts[f"{detector.value}.fp"] += 1
                        elif expected_hit and not predicted_hit:
                            counts[f"{detector.value}.fn"] += 1
                        else:
                            counts[f"{detector.value}.tn"] += 1
                else:
                    unlabelled_runs += 1

                latency = {}
                if labels_status == "labelled":
                    for detector in sorted(expected, key=lambda item: item.value):
                        if detector in first_detection_index:
                            value = first_detection_index[detector] + 1
                            latency[detector.value] = value
                            latency_samples[detector.value].append(value)
                        else:
                            latency[detector.value] = None

                run_results.append({
                    "scenario": name,
                    "run_id": run_id,
                    "labels_status": labels_status,
                    "expected": sorted(item.value for item in expected),
                    "predicted": sorted(item.value for item in predicted),
                    "detection_latency_events": latency,
                })
                monitor.end_run(run_id)

            monitor.close()

    metrics = {}
    for detector in DetectorType:
        tp = counts[f"{detector.value}.tp"]
        fp = counts[f"{detector.value}.fp"]
        fn = counts[f"{detector.value}.fn"]
        tn = counts[f"{detector.value}.tn"]
        samples = latency_samples[detector.value]
        # A metric is undefined when its denominator is zero. In particular,
        # recall/F1 must not look like a measured zero when the reviewed set
        # contains no positive examples for that detector.
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1_denominator = 2 * tp + fp + fn
        f1 = 2 * tp / f1_denominator if f1_denominator else None
        fpr = fp / (fp + tn) if fp + tn else None

        metrics[detector.value] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "mean_detection_latency_events": round(sum(samples) / len(samples), 4) if samples else None,
            "max_detection_latency_events": max(samples) if samples else None,
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "false_positive_rate": round(fpr, 4) if fpr is not None else None,
        }

    return {
        "scenarios": len(scenarios),
        "runs": len(run_results),
        "labelled_runs": labelled_runs,
        "unlabelled_runs": unlabelled_runs,
        "metrics": metrics,
        "runs_detail": run_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate labelled DriftShield trace data")
    parser.add_argument("dataset", type=Path, help="Path to labelled JSON dataset")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    report = validate_dataset(args.dataset)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
