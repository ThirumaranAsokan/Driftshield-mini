"""Convert independently reviewed SWE-agent records to DriftShield validation JSON."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def text_of(item: dict[str, Any]) -> str:
    return str(item.get("text") or item.get("content") or "")


def role_of(item: dict[str, Any]) -> str:
    return str(item.get("role", "")).lower()


def code_commands(text: str) -> list[str]:
    commands = []
    for block in re.findall(r"\`\`\`(?:[^\n]*)\n(.*?)\`\`\`", text, re.DOTALL):
        first = next((line.strip() for line in block.splitlines() if line.strip()), "")
        if first:
            commands.append(first[:160])
    return commands


def extract_goal(text: str) -> str:
    if "ISSUE:" in text:
        goal = text.split("ISSUE:", 1)[1]
        if "INSTRUCTIONS:" in goal:
            goal = goal.split("INSTRUCTIONS:", 1)[0]
        return goal.strip()
    return text.strip()


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

        if role in {"ai", "assistant", "model"}:
            for command in code_commands(text):
                events.append(
                    {
                        "action_type": "tool_call",
                        "action_name": command.split()[0].lower()[:100],
                        "token_count": max(1, len(command) // 4),
                        "duration_ms": 0,
                        "input_data": {"command": command},
                        "output_data": {},
                        "metadata": {"source_turn": index},
                    }
                )

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
        name
        for name in ("action_loop", "goal_drift", "resource_spike")
        if labels.get(name) is True
    ]
    goal = ""
    for item in trajectory:
        if isinstance(item, dict) and role_of(item) == "user":
            goal = extract_goal(text_of(item))
            if goal:
                break

    return {
        "run_id": str(row["review_id"]),
        "expected_detectors": expected,
        "events": events,
        "review_confidence": row.get("review_confidence", ""),
        "goal_source": "ISSUE section of first user trajectory turn",
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
                "independently_reviewed": False,
                "review_method": payload.get("review_method"),
                "scenarios": scenarios,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
