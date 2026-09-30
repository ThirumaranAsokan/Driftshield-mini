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
the dataset author. Results are engineering measurements, not production
accuracy claims.
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
    run_results = []

    with tempfile.TemporaryDirectory(prefix="driftshield-validation-") as tmp:
        for scenario in scenarios:
            name = str(scenario["name"])
            monitor = DriftMonitor(
                agent_id=name,
                db_path=str(Path(tmp) / f"{name}.db"),
                goal_description=str(scenario.get("goal", "")),
                calibration_runs=int(scenario.get("calibration_runs", 30)),
                loop_max_repeats=int(scenario.get("loop_max_repeats", 4)),
            )

            for run in scenario["runs"]:
                run_id = monitor.start_run(run_id=str(run["run_id"]))
                predicted: set[DetectorType] = set()

                for event in run["events"]:
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
                    predicted.update(item.detector for item in detected)

                expected = _detectors(list(run.get("expected_detectors", [])))
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

                run_results.append({
                    "scenario": name,
                    "run_id": run_id,
                    "expected": sorted(item.value for item in expected),
                    "predicted": sorted(item.value for item in predicted),
                })
                monitor.end_run(run_id)

            monitor.close()

    metrics = {}
    for detector in DetectorType:
        tp = counts[f"{detector.value}.tp"]
        fp = counts[f"{detector.value}.fp"]
        fn = counts[f"{detector.value}.fn"]
        tn = counts[f"{detector.value}.tn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        metrics[detector.value] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "false_positive_rate": round(fpr, 4),
        }

    return {
        "scenarios": len(scenarios),
        "runs": len(run_results),
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
