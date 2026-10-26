"""Convert independently reviewed FinTrace records to DriftShield validation JSON.

Only records with a resolved positive/negative label for every detector are
included. Blank or uncertain labels are excluded, never treated as negatives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DETECTORS = ("action_loop", "goal_drift", "resource_spike")
POSITIVE = {"positive", "pos", "yes", "true", "1"}
NEGATIVE = {"negative", "neg", "no", "false", "0"}
UNCERTAIN = {"uncertain", "unknown", "unsure", "n/a", "na", "null", "none", ""}


def _label(value: Any) -> bool | None:
    """Normalize supported reviewer labels; return None for unresolved labels."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in POSITIVE:
        return True
    if normalized in NEGATIVE:
        return False
    if normalized in UNCERTAIN:
        return None
    raise ValueError(f"Unsupported review label: {value!r}")


def _to_events(excerpt: Any) -> list[dict[str, Any]]:
    if not isinstance(excerpt, list):
        raise TypeError("trajectory_excerpt must be a list")
    events = []
    for index, item in enumerate(excerpt):
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type", "")).lower()
        role = str(item.get("role", "")).lower()
        if kind == "function_call":
            action_type = "tool_call"
            action_name = str(item.get("tool_name") or "unknown_tool")
            input_data = {"arguments": item.get("arguments", "")}
            output_data = {}
        elif kind == "function_call_output":
            action_type = "tool_result"
            action_name = str(item.get("tool_name") or "tool_result")
            input_data = {}
            output_data = {"result": item.get("content", "")}
        elif kind == "message" or role in {"assistant", "ai", "model"}:
            action_type = "llm_request"
            action_name = "assistant_output"
            input_data = {}
            output_data = {"text": str(item.get("content", ""))}
        else:
            # Preserve user/system messages as context, not detector outputs.
            continue
        events.append({
            "action_type": action_type,
            "action_name": action_name,
            "token_count": 0,
            "duration_ms": 0,
            "input_data": input_data,
            "output_data": output_data,
            "metadata": {
                "source_turn": item.get("turn", index),
                "source_type": kind or role,
            },
        })
    return events


def convert_records(payload: dict[str, Any], split: str) -> dict[str, Any]:
    if split not in {"calibration", "holdout"}:
        raise ValueError("split must be calibration or holdout")
    rows = payload.get("runs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Review file must contain a non-empty 'runs' list")

    scenarios = []
    excluded: list[dict[str, str]] = []
    included = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("split", "")).strip().lower() != split:
            continue
        review_id = str(row.get("review_id", "")).strip()
        if not review_id:
            raise ValueError("Every reviewed record must have a review_id")
        labels = row.get("labels")
        if not isinstance(labels, dict):
            raise TypeError(f"Review {review_id} is missing labels")

        normalized = {name: _label(labels.get(name)) for name in DETECTORS}
        unresolved = [name for name, value in normalized.items() if value is None]
        if unresolved:
            excluded.append({"review_id": review_id, "reason": "unresolved labels: " + ", ".join(unresolved)})
            continue

        expected = [name for name, value in normalized.items() if value is True]
        events = _to_events(row.get("trajectory_excerpt", []))
        scenarios.append({
            "name": "fintrace-" + review_id,
            "goal": str(row.get("task_goal", "")),
            "calibration_runs": 0 if split == "holdout" else 1,
            "loop_max_repeats": 4,
            "runs": [{
                "run_id": review_id,
                "expected_detectors": expected,
                "events": events,
                "metadata": {
                    "source": "YupengCao/FinTrace",
                    "review_id": review_id,
                    "split": split,
                    "labels_status": "labelled",
                    "reviewer_id": row.get("reviewer_id", ""),
                    "review_confidence": row.get("review_confidence", ""),
                },
            }],
        })
        included += 1

    if included == 0:
        raise ValueError(f"No fully labelled records found for split {split!r}")
    return {
        "schema_version": 1,
        "source": "YupengCao/FinTrace",
        "split": split,
        "label_provenance": "review labels supplied by input file; independence is not verified by this converter",
        "included_runs": included,
        "excluded_unresolved_count": len(excluded),
        "excluded_unresolved": excluded,
        "scenarios": scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("calibration", "holdout"), required=True)
    args = parser.parse_args()

    payload = json.loads(args.review_file.read_text(encoding="utf-8"))
    converted = convert_records(payload, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(converted, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"Converted {converted['included_runs']} labelled {args.split} runs; "
        f"excluded {converted['excluded_unresolved_count']} with unresolved labels."
    )


if __name__ == "__main__":
    main()
