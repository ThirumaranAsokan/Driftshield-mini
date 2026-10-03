"""Convert independently reviewed SWE-agent records to DriftShield validation JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def text_of(item: dict[str, Any]) -> str:
    return str(item.get("text") or item.get("content") or "")


def role_of(item: dict[str, Any]) -> str:
    return str(item.get("role", "")).lower()


def action_name(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        lower = line.lower()
        if lower.startswith(("action:", "command:")):
            return line.split(":", 1)[1].strip().split()[0].lower()[:100]
    marker = "will execute following command for "
    lower = text.lower()
    start = lower.find(marker)
    if start >= 0:
        rest = text[start + len(marker):]
        return rest.split(":", 1)[0].strip().lower()[:100] or None
    return None


def to_run(row: dict[str, Any]) -> dict[str, Any]:
    events = []
    trajectory = row.get("trajectory", [])
    if isinstance(trajectory, str):
        trajectory = json.loads(trajectory)

    for index, item in enumerate(trajectory):
        if not isinstance(item, dict):
            continue
        text = text_of(item)
        if not text:
            continue
        role = role_of(item)
        action = action_name(text)
        if action:
            events.append(
                {
                    "action_type": "tool_call",
                    "action_name": action,
                    "token_count": max(1, len(text) // 4),
                    "duration_ms": 0,
                    "input_data": {},
                    "output_data": {},
                    "metadata": {"source_turn": index},
                }
            )
        elif role in {"ai", "assistant", "model"}:
            events.append(
                {
                    "action_type": "llm_request",
                    "action_name": "assistant_output",
                    "token_count": max(1, len(text) // 4),
                    "duration_ms": 0,
                    "input_data": {},
                    "output_data": {"text": text},
                    "metadata": {"source_turn": index},
                }
            )

    labels = row["labels"]
    expected = [
        name for name in ("action_loop", "goal_drift", "resource_spike")
        if labels.get(name) is True
    ]
    goal = ""
    for item in trajectory:
        if isinstance(item, dict) and role_of(item) == "user":
            goal = text_of(item)
            if goal:
                break

    return {
        "run_id": str(row["review_id"]),
        "expected_detectors": expected,
        "events": events,
        "review_confidence": row.get("review_confidence", ""),
        "goal_source": "first user trajectory turn",
        "goal": goal,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("review_file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("calibration", "holdout"), required=True)
    args = parser.parse_args()

    payload = json.loads(args.review_file.read_text(encoding="utf-8"))
    rows = payload.get("runs", [])
    if not rows:
        raise SystemExit("No reviewed runs found")
    if any(any(v is None for v in row.get("labels", {}).values()) for row in rows):
        raise SystemExit("All detector labels must be frozen before conversion")

    converted = [to_run(row) for row in rows]
    scenarios = []
    for row in converted:
        scenarios.append(
            {
                "name": row["run_id"],
                "goal": row["goal"],
                "calibration_runs": 0 if args.split == "holdout" else 1,
                "loop_max_repeats": 4,
                "runs": [row],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "nebius/SWE-agent-trajectories",
                "split": args.split,
                "independently_reviewed": True,
                "scenarios": scenarios,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
