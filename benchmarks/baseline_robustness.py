"""Controlled baseline robustness experiment.

This is an engineering diagnostic, not a production accuracy benchmark.
It compares stable, contaminated, and step-change workloads and prints the
calibrated baseline plus whether a follow-up run is flagged.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from driftshield_mini import DriftMonitor
from driftshield_mini.models import DetectorType


SCENARIOS = {
    "stable": [100, 100, 100, 100],
    "contaminated": [100, 100, 100, 1000],
    "step_change": [100, 100, 100, 300],
}


def run(name: str, calibration: list[int], follow_up: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="driftshield-baseline-") as tmp:
        monitor = DriftMonitor(
            agent_id=name,
            db_path=str(Path(tmp) / "baseline.db"),
            calibration_runs=len(calibration),
        )
        for index, tokens in enumerate(calibration):
            run_id = monitor.start_run(run_id=f"cal-{index}")
            monitor.record_event(
                "llm_request", "work", run_id=run_id, token_count=tokens, duration_ms=10
            )
            monitor.end_run(run_id)

        baseline = monitor.get_baseline()
        run_id = monitor.start_run(run_id="follow-up")
        events = monitor.record_event(
            "llm_request", "work", run_id=run_id, token_count=follow_up, duration_ms=10
        )
        monitor.end_run(run_id)
        monitor.close()

        detected = sorted(
            event.detector.value
            for event in events
            if event.detector == DetectorType.RESOURCE_SPIKE
        )
        return {
            "scenario": name,
            "calibration_tokens": calibration,
            "follow_up_tokens": follow_up,
            "mean_tokens": baseline.mean_tokens_per_run if baseline else None,
            "std_tokens": baseline.std_tokens_per_run if baseline else None,
            "resource_spike_detected": bool(detected),
        }


def main() -> None:
    results = [
        run("stable", SCENARIOS["stable"], 200),
        run("contaminated", SCENARIOS["contaminated"], 200),
        run("step_change", SCENARIOS["step_change"], 400),
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
