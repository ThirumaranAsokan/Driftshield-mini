"""Create a deterministic, blinded human-review set from FinTrace trajectories.

Only task prompts and observed output trajectories are included. Reference answers,
golden trajectories, benchmark scores, and DriftShield predictions are excluded.
No detector labels are inferred by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any


DETECTORS = ("action_loop", "goal_drift", "resource_spike")


def iter_rows(payload: Any) -> Iterable[dict[str, Any]]:
    """Accept a top-level list or a common data/rows JSON envelope."""
    if isinstance(payload, list):
        yield from (row for row in payload if isinstance(row, dict))
    elif isinstance(payload, dict):
        rows = payload.get("data", payload.get("rows", []))
        if isinstance(rows, list):
            yield from (row for row in rows if isinstance(row, dict))


def stable_source_id(row: dict[str, Any]) -> str:
    """Return a stable identifier without exposing content in the review sheet."""
    identity = row.get("id")
    if identity not in (None, ""):
        raw = f"fintrace|{identity}"
    else:
        raw = json.dumps(
            {
                "source_query": row.get("source_query", ""),
                "output_trajectory": row.get("output_trajectory", []),
            },
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
    return hashlib.sha256(str(raw).encode("utf-8")).hexdigest()[:16]


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        pieces = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text", item.get("content", ""))
                if text:
                    pieces.append(str(text))
            elif item:
                pieces.append(str(item))
        return "\n".join(pieces)
    return str(value or "")


def trajectory_excerpt(raw: Any, max_messages: int = 80) -> list[dict[str, Any]]:
    """Select observed messages/tool calls only; omit reference and scoring fields."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return [{"role": "unknown", "content": raw[:4000]}] if raw else []
    if not isinstance(raw, list):
        return []

    entries: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        # FinTrace's turn/output schema contains function_call and message records.
        if isinstance(item.get("output"), list) and "role" not in item:
            turn_number = item.get("turn")
            for output in item["output"]:
                if not isinstance(output, dict):
                    continue
                kind = output.get("type", "unknown")
                if kind == "function_call":
                    entry = {
                        "turn": turn_number,
                        "type": kind,
                        "tool_name": str(output.get("name", "")),
                        "arguments": _content_text(output.get("arguments", ""))[:2000],
                    }
                elif kind == "message":
                    entry = {
                        "turn": turn_number,
                        "type": kind,
                        "role": str(output.get("role", "")),
                        "content": _content_text(output.get("content", ""))[:4000],
                    }
                elif kind == "function_call_output":
                    entry = {
                        "turn": turn_number,
                        "type": kind,
                        "tool_name": str(output.get("name", "")),
                        "content": _content_text(output.get("output", output.get("content", "")))[:2000],
                    }
                else:
                    continue
                entries.append(entry)
                if len(entries) >= max_messages:
                    return entries
        else:
            role = str(item.get("role", "unknown"))
            content = _content_text(item.get("content", item.get("text", "")))
            entry: dict[str, Any] = {"role": role, "content": content[:4000]}
            calls = item.get("tool_calls")
            if isinstance(calls, list):
                entry["tool_calls"] = [
                    {
                        "tool_name": str(
                            call.get("function", {}).get("name", call.get("name", ""))
                            if isinstance(call.get("function", {}), dict)
                            else call.get("name", "")
                        ),
                        "arguments": _content_text(
                            call.get("function", {}).get("arguments", call.get("arguments", ""))
                            if isinstance(call.get("function", {}), dict)
                            else call.get("arguments", "")
                        )[:2000],
                    }
                    for call in calls
                    if isinstance(call, dict)
                ]
            entries.append(entry)
            if len(entries) >= max_messages:
                return entries
    return entries


def sample_rows(rows: Iterable[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    """Deterministic reservoir sample, independent of detector predictions."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        item = dict(row)
        if index < limit:
            reservoir.append(item)
            continue
        slot = rng.randrange(index + 1)
        if slot < limit:
            reservoir[slot] = item
    rng.shuffle(reservoir)
    return reservoir


def build_review_records(
    rows: Iterable[dict[str, Any]],
    limit: int = 300,
    seed: int = 20261015,
    calibration_count: int = 200,
    max_messages: int = 80,
) -> list[dict[str, Any]]:
    if calibration_count < 0 or calibration_count > limit:
        raise ValueError("calibration_count must be between 0 and limit")
    selected = sample_rows(rows, limit, seed)
    records = []
    for index, row in enumerate(selected):
        source_id = stable_source_id(row)
        labels = {detector: "" for detector in DETECTORS}
        reasons = {detector: "" for detector in DETECTORS}
        records.append(
            {
                "review_id": source_id,
                "source": "YupengCao/FinTrace",
                "source_id": str(row.get("id", source_id)),
                "split": "calibration" if index < calibration_count else "holdout",
                "task_type": str(row.get("task_type", "")),
                "task_goal": _content_text(row.get("source_query", ""))[:4000],
                "trajectory_excerpt": trajectory_excerpt(
                    row.get("output_trajectory", []), max_messages=max_messages
                ),
                "labels": labels,
                "review_reason": reasons,
                "reviewer_id": "",
                "review_confidence": "",
                "adjudication_needed": "",
            }
        )
    return records


def load_rows(input_path: Path | None) -> Iterable[dict[str, Any]]:
    if input_path:
        yield from iter_rows(json.loads(input_path.read_text(encoding="utf-8")))
        return
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            'Install external validation first: python -m pip install -e ".[external-validation]"'
        ) from exc
    yield from (dict(row) for row in load_dataset("YupengCao/FinTrace", split="test", streaming=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Local JSON copy; otherwise load FinTrace test split")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--calibration-count", type=int, default=200)
    parser.add_argument("--max-messages", type=int, default=80)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.max_messages <= 0:
        parser.error("--max-messages must be positive")
    records = build_review_records(
        load_rows(args.input),
        limit=args.limit,
        seed=args.seed,
        calibration_count=args.calibration_count,
        max_messages=args.max_messages,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "dataset": "YupengCao/FinTrace",
        "purpose": "independent DriftShield detector labelling",
        "label_status": "unlabelled",
        "sampling": {
            "method": "deterministic reservoir sampling over the supplied dataset split",
            "seed": args.seed,
            "requested_size": args.limit,
            "actual_size": len(records),
            "calibration_count": sum(record["split"] == "calibration" for record in records),
            "holdout_count": sum(record["split"] == "holdout" for record in records),
        },
        "blinding": (
            "DriftShield predictions, FinTrace reference answers, golden trajectories, "
            "and task outcome scores are intentionally excluded."
        ),
        "reviewer_instructions": "See docs/fintrace-external-analysis.md",
        "runs": records,
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Created {len(records)} unlabelled FinTrace review records: {args.output}")


if __name__ == "__main__":
    main()
