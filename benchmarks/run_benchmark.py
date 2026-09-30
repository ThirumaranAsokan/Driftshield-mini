"""Deterministic labelled benchmark for DriftShield Mini detectors.

This benchmark measures detector classification against labelled synthetic traces.
It does not claim production accuracy; representative customer traces are still
required for that.
"""

from __future__ import annotations

import json
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from driftshield_mini import DriftMonitor
from driftshield_mini.models import DetectorType


class FakeEmbedder:
    """Deterministic stand-in so the benchmark never downloads a model."""

    def encode(self, text: str):
        text = str(text).lower()
        return [1.0, 0.0] if any(word in text for word in ("financial", "finance", "report")) else [0.0, 1.0]


@dataclass(frozen=True)
class Case:
    name: str
    expected: frozenset[DetectorType]
    actions: tuple[str, ...] = ()
    goal: str = ""
    output: str = ""
    token_count: int = 0


CASES = (
    Case("true_single_loop", frozenset({DetectorType.ACTION_LOOP}), ("search",) * 6),
    Case("true_cycle", frozenset({DetectorType.ACTION_LOOP}), ("search", "format") * 4),
    Case("legitimate_repetition", frozenset(), ("search",) * 3),
    Case("legitimate_sequence", frozenset(), ("search", "format") * 2),
    Case(
        "goal_drift",
        frozenset({DetectorType.GOAL_DRIFT}),
        goal="Summarise financial reports",
        output="Explain how to grow tomatoes in a garden.",
    ),
    Case(
        "goal_preserving",
        frozenset(),
        goal="Summarise financial reports",
        output="The financial report shows revenue increased during the quarter.",
    ),
    Case(
        "goal_paraphrase",
        frozenset(),
        goal="Summarise financial reports",
        output="The quarterly finance report shows stronger revenue.",
    ),
    Case("resource_spike", frozenset({DetectorType.RESOURCE_SPIKE}), token_count=60_000),
    Case("normal_resource_use", frozenset(), token_count=1_000),
    Case(
        "combined_loop_and_resource",
        frozenset({DetectorType.ACTION_LOOP, DetectorType.RESOURCE_SPIKE}),
        actions=("search",) * 6,
        token_count=60_000,
    ),
    Case("normal_mixed_workflow", frozenset(), actions=("search", "format", "save")),
    Case(
        "unrelated_goal_output",
        frozenset({DetectorType.GOAL_DRIFT}),
        goal="Summarise financial reports",
        output="Explain how to repair a bicycle safely and efficiently.",
    ),
)


def run_case(case: Case) -> tuple[set[DetectorType], dict[DetectorType, int]]:
    with tempfile.TemporaryDirectory() as tmp:
        monitor = DriftMonitor(
            agent_id=case.name,
            db_path=str(Path(tmp) / "trace.db"),
            goal_description=case.goal,
            calibration_runs=999,
            loop_max_repeats=4,
        )
        run_id = monitor.start_run()
        first_detection: dict[DetectorType, int] = {}

        events = list(case.actions)
        if case.output:
            events.append("__goal_output__")
        if case.token_count and not events:
            events.append("__resource__")

        for index, action in enumerate(events, start=1):
            if action == "__goal_output__":
                detected = monitor.record_event(
                    "llm_request",
                    "agent_complete",
                    run_id=run_id,
                    input_data={"input": case.goal},
                    output_data={"text": case.output},
                )
            else:
                detected = monitor.record_event(
                    "tool_call",
                    action,
                    run_id=run_id,
                    token_count=case.token_count if index == len(events) else 0,
                )
            for event in detected:
                first_detection.setdefault(event.detector, index)

        detected = {
            event.detector
            for event in monitor.get_recent_alerts(hours=1, limit=100)
        }
        monitor.end_run(run_id)
        monitor.close()
        return detected, first_detection


def classification_metrics(cases: tuple[Case, ...], predictions: dict[str, set[DetectorType]]):
    metrics = {}
    for detector in DetectorType:
        tp = fp = fn = tn = 0
        for case in cases:
            expected = detector in case.expected
            predicted = detector in predictions[case.name]
            if expected and predicted:
                tp += 1
            elif not expected and predicted:
                fp += 1
            elif expected and not predicted:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        metrics[detector.value] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "false_positive_rate": round(fpr, 4),
        }
    return metrics


def measure_overhead(repetitions: int = 3, events_per_run: int = 100) -> dict[str, float]:
    baseline_times = []
    monitored_times = []

    for _ in range(repetitions):
        started = time.perf_counter()
        for index in range(events_per_run):
            _ = {"action_type": "tool_call", "action_name": f"tool_{index % 5}"}
        baseline_times.append(time.perf_counter() - started)

        with tempfile.TemporaryDirectory() as tmp:
            monitor = DriftMonitor(
                agent_id="overhead",
                db_path=str(Path(tmp) / "trace.db"),
                calibration_runs=999,
                loop_max_repeats=1000,
            )
            run_id = monitor.start_run()
            started = time.perf_counter()
            for index in range(events_per_run):
                monitor.record_event(
                    "tool_call",
                    f"tool_{index % 5}",
                    run_id=run_id,
                )
            monitored_times.append(time.perf_counter() - started)
            monitor.close()

    baseline = statistics.median(baseline_times)
    monitored = statistics.median(monitored_times)
    overhead = ((monitored / baseline) - 1.0) * 100 if baseline else 0.0
    return {
        "events_per_run": events_per_run,
        "baseline_ms": round(baseline * 1000, 3),
        "monitored_ms": round(monitored * 1000, 3),
        "estimated_overhead_percent": round(overhead, 2),
    }


def main() -> None:
    import driftshield_mini.detectors.goal_drift as goal_drift

    original_loader = goal_drift.load_embedding_model
    goal_drift.load_embedding_model = lambda: FakeEmbedder()
    try:
        predictions = {}
        latencies = {}
        for case in CASES:
            detected, first_detection = run_case(case)
            predictions[case.name] = detected
            latencies[case.name] = first_detection

        result = {
            "case_count": len(CASES),
            "metrics": classification_metrics(CASES, predictions),
            "detection_latency_events": latencies,
            "monitoring_overhead": measure_overhead(),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        goal_drift.load_embedding_model = original_loader


if __name__ == "__main__":
    main()
