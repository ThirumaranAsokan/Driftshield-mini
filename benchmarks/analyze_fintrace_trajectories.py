"""Explore DriftShield signals on the public FinTrace financial trajectories.

This is external behavioural analysis only. FinTrace's own evaluation labels are
not used as DriftShield detector labels.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from driftshield_mini import DriftMonitor
from driftshield_mini.models import DetectorType


def _as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "") or item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content or "")


def _tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = message.get("tool_calls", [])
    if isinstance(calls, list):
        return [call for call in calls if isinstance(call, dict)]
    return []


def _tool_name(call: dict[str, Any]) -> str:
    function = call.get("function", {})
    if isinstance(function, dict):
        return str(function.get("name") or call.get("name") or "unknown_tool")
    return str(call.get("name") or "unknown_tool")


def _tool_arguments(call: dict[str, Any]) -> str:
    function = call.get("function", {})
    if isinstance(function, dict):
        return _as_text(function.get("arguments", ""))
    return _as_text(call.get("arguments", ""))


def _message_token_estimate(message: dict[str, Any]) -> int:
    text = _as_text(message.get("content", ""))
    calls = _tool_calls(message)
    text += "".join(_tool_arguments(call) for call in calls)
    return max(0, len(text) // 4)


def iter_rows(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        yield from (row for row in payload if isinstance(row, dict))
    elif isinstance(payload, dict):
        rows = payload.get("data", payload.get("rows", []))
        if isinstance(rows, list):
            yield from (row for row in rows if isinstance(row, dict))


def analyse_row(
    row: dict[str, Any],
    loop_repeats: int,
    token_limit: int,
) -> dict[str, Any]:
    trajectory = row.get("output_trajectory", [])
    if isinstance(trajectory, str):
        trajectory = json.loads(trajectory)
    if not isinstance(trajectory, list):
        trajectory = []

    goal = str(row.get("source_query", ""))
    monitor = DriftMonitor(
        agent_id=f"fintrace-{row.get('id', 'unknown')}",
        db_path=":memory:",
        goal_description=goal,
        calibration_runs=0,
        loop_max_repeats=loop_repeats,
    )
    run_id = monitor.start_run(run_id=f"fintrace-{row.get('id', 'unknown')}")

    counts = Counter()
    action_count = 0
    token_estimate = 0
    first_detection: dict[str, int] = {}

    try:
        for index, message in enumerate(trajectory, start=1):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).lower()
            token_estimate += _message_token_estimate(message)

            for call_index, call in enumerate(_tool_calls(message)):
                name = _tool_name(call)
                arguments = _tool_arguments(call)
                detected = monitor.record_event(
                    "tool_call",
                    name,
                    run_id=run_id,
                    token_count=max(0, len(arguments) // 4),
                    input_data={"arguments": arguments[:4000]},
                    metadata={
                        "source": "FinTrace",
                        "trajectory_message": index,
                        "tool_call_index": call_index,
                        "call_id": call.get("id", call.get("call_id")),
                    },
                )
                action_count += 1
                for event in detected:
                    detector = event.detector.value
                    counts[detector] += 1
                    first_detection.setdefault(detector, index)

            if role in {"assistant", "model"}:
                text = _as_text(message.get("content", "")).strip()
                if text:
                    detected = monitor.record_event(
                        "llm_request",
                        "assistant_output",
                        run_id=run_id,
                        token_count=max(0, len(text) // 4),
                        output_data={"text": text[:8000]},
                        metadata={
                            "source": "FinTrace",
                            "trajectory_message": index,
                        },
                    )
                    for event in detected:
                        detector = event.detector.value
                        counts[detector] += 1
                        first_detection.setdefault(detector, index)

        resource_signal = token_estimate > token_limit
        if resource_signal:
            counts[DetectorType.RESOURCE_SPIKE.value] += 1

        return {
            "id": row.get("id"),
            "task_type": row.get("task_type"),
            "trajectory_messages": len(trajectory),
            "tool_actions": action_count,
            "estimated_tokens": token_estimate,
            "signals": dict(counts),
            "first_detection_message": first_detection,
            "resource_threshold_exceeded": resource_signal,
        }
    finally:
        monitor.end_run(run_id)
        monitor.close()


def analyse_rows(
    rows: Iterable[dict[str, Any]],
    loop_repeats: int = 4,
    token_limit: int = 50000,
    limit: int | None = None,
) -> dict[str, Any]:
    results = []
    for index, row in enumerate(rows):
        if limit is not None and index >= limit:
            break
        results.append(analyse_row(row, loop_repeats, token_limit))

    totals = Counter()
    for result in results:
        totals.update(result["signals"])

    return {
        "dataset": "YupengCao/FinTrace",
        "rows_analyzed": len(results),
        "loop_repeats": loop_repeats,
        "resource_token_threshold": token_limit,
        "signals": dict(totals),
        "results": results,
        "accuracy_metrics": "not computed: FinTrace does not provide DriftShield detector labels",
        "golden_trajectory_note": (
            "golden_trajectories are evaluation material, not DriftShield detector labels"
        ),
        "resource_note": (
            "Resource signals use estimated trajectory text/tool-argument tokens, "
            "not provider usage telemetry."
        ),
    }


def load_rows() -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Install the external validation extra first: "
            'python -m pip install -e ".[external-validation]"'
        ) from exc

    dataset = load_dataset("YupengCao/FinTrace", split="test")
    return (dict(row) for row in dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--loop-repeats", type=int, default=4)
    parser.add_argument("--token-limit", type=int, default=50000)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = (
        iter_rows(json.loads(args.input.read_text(encoding="utf-8")))
        if args.input
        else load_rows()
    )
    result = analyse_rows(rows, args.loop_repeats, args.token_limit, args.limit)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
