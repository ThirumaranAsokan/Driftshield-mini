"""Collect Finance Agent ATIF files into one DriftShield validation dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.ingest_atif import convert_trajectory


def collect(root: Path) -> dict[str, Any]:
    files = sorted(root.rglob("trajectory_atif.json"))
    if not files:
        raise ValueError(f"No trajectory_atif.json files found under {root}")

    runs: list[dict[str, Any]] = []
    scenario_name = "finance-agent"
    goal = ""

    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        converted = convert_trajectory(payload, path.name)
        scenario = converted["scenarios"][0]
        scenario_name = scenario["name"] or scenario_name
        goal = goal or scenario["goal"]
        runs.extend(scenario["runs"])

    return {
        "scenarios": [
            {
                "name": scenario_name,
                "goal": goal,
                "calibration_runs": 30,
                "runs": runs,
            }
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Finance Agent ATIF trajectories")
    parser.add_argument("root", type=Path, help="Finance Agent logs directory")
    parser.add_argument("--output", type=Path, required=True, help="Combined validation JSON")
    args = parser.parse_args()

    dataset = collect(args.root)
    args.output.write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    print(f"Collected {len(dataset['scenarios'][0]['runs'])} runs into {args.output}")


if __name__ == "__main__":
    main()
