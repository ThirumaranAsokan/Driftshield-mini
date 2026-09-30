"""Controlled local measurement of monitoring overhead.

Results are environment-specific engineering measurements, not production claims.
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from driftshield_mini import DriftMonitor

EVENTS = 200
REPEATS = 5


def workload_without_monitor() -> None:
    for index in range(EVENTS):
        payload = {"action_name": "search" if index % 3 else "format", "token_count": 100 + index}
        _ = payload["action_name"]


def workload_with_monitor(db_path: str) -> int:
    monitor = DriftMonitor(agent_id="overhead", db_path=db_path, calibration_runs=9999)
    run_id = monitor.start_run(run_id="overhead-run")
    for index in range(EVENTS):
        monitor.record_event(
            "tool_call", "search" if index % 3 else "format", run_id=run_id,
            token_count=100 + index, duration_ms=5.0,
        )
    monitor.end_run(run_id)
    monitor.close()
    return Path(db_path).stat().st_size


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def main() -> None:
    baseline_samples, monitored_samples, storage = [], [], []
    for _ in range(REPEATS):
        start = time.perf_counter()
        workload_without_monitor()
        baseline_samples.append(time.perf_counter() - start)
        with tempfile.TemporaryDirectory(prefix="driftshield-overhead-") as tmp:
            db_path = str(Path(tmp) / "trace.db")
            start = time.perf_counter()
            size = workload_with_monitor(db_path)
            monitored_samples.append(time.perf_counter() - start)
            storage.append(float(size))
    baseline, monitored = median(baseline_samples), median(monitored_samples)
    result = {
        "events": EVENTS, "repeats": REPEATS,
        "baseline_seconds_median": round(baseline, 6),
        "monitored_seconds_median": round(monitored, 6),
        "relative_overhead_percent": round(((monitored / baseline) - 1) * 100, 2) if baseline else None,
        "baseline_events_per_second": round(EVENTS / baseline, 2) if baseline else None,
        "monitored_events_per_second": round(EVENTS / monitored, 2) if monitored else None,
        "sqlite_bytes_median": round(median(storage)),
        "note": "Synthetic local workload; not a production performance claim.",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
