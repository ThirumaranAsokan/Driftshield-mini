"""Convert ATIF v1.7 agent trajectories into DriftShield validation traces.

This adapter is intentionally label-neutral. It preserves telemetry supplied by the
agent runtime and never infers action_loop, goal_drift, or resource_spike labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _observation_map(step: dict[str, Any]) -> dict[str, Any]:
    observation = step.get("observation") or {}
    return {
        str(item["source_call_id"]): item.get("content", "")
        for item in observation.get("results", [])
        if isinstance(item, dict) and "source_call_id" in item
    }


def _goal(steps: list[dict[str, Any]]) -> str:
    for step in steps:
        if step.get("source") == "user":
            return str(step.get("message", ""))
    return ""


def _event(
    *,
    action_type: str,
    action_name: str,
    step: dict[str, Any],
    token_count: int = 0,
    duration_ms: float = 0.0,
    input_data: dict[str, Any] | None = None,
    output_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "action_type": action_type,
        "action_name": action_name,
        "token_count": int(token_count),
        "duration_ms": float(duration_ms),
        "input_data": input_data or {},
        "output_data": output_data or {},
        "metadata": {
            "atif_step_id": step.get("step_id"),
            "timestamp": step.get("timestamp"),
            "model_name": step.get("model_name"),
        },
    }


def convert_trajectory(payload: dict[str, Any], source_name: str = "") -> dict[str, Any]:
    if payload.get("schema_version") != "ATIF-v1.7":
        raise ValueError("Expected ATIF-v1.7 trajectory")

    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("ATIF trajectory must contain a non-empty steps list")

    session_id = str(payload.get("session_id") or "atif-run")
    agent = payload.get("agent") or {}
    goal = _goal(steps)
    observations = {step.get("step_id"): _observation_map(step) for step in steps}

    events: list[dict[str, Any]] = []
    for step in steps:
        source = step.get("source")
        if source == "agent":
            metrics = step.get("metrics") or {}
            token_count = int(metrics.get("prompt_tokens", 0)) + int(
                metrics.get("completion_tokens", 0)
            )
            extra = metrics.get("extra") or {}
            duration_ms = float(extra.get("duration_seconds", 0.0)) * 1000.0

            events.append(
                _event(
                    action_type="llm_request",
                    action_name="agent_turn",
                    step=step,
                    token_count=token_count,
                    duration_ms=duration_ms,
                    input_data={
                        "message": step.get("message", ""),
                        "reasoning": step.get("reasoning_content"),
                    },
                    output_data={"model_name": step.get("model_name")},
                )
            )

            step_observations = observations.get(step.get("step_id"), {})
            for call in step.get("tool_calls") or []:
                call_id = str(call.get("tool_call_id", ""))
                result = step_observations.get(call_id, "")
                events.append(
                    _event(
                        action_type="tool_call",
                        action_name=str(call.get("function_name", "unknown_tool")),
                        step=step,
                        input_data=dict(call.get("arguments") or {}),
                        output_data={"result": result},
                    )
                )

        elif source == "system":
            events.append(
                _event(
                    action_type="state_transition",
                    action_name="system_message",
                    step=step,
                    output_data={"message": step.get("message", "")},
                )
            )

    final_metrics = payload.get("final_metrics") or {}
    return {
        "scenarios": [
            {
                "name": str(agent.get("name") or "real-agent"),
                "goal": goal,
                "calibration_runs": 30,
                "runs": [
                    {
                        "run_id": session_id,
                        "expected_detectors": [],
                        "events": events,
                        "metadata": {
                            "source_format": "ATIF-v1.7",
                            "source_file": source_name,
                            "model_name": agent.get("model_name"),
                            "final_metrics": final_metrics,
                            "trajectory_extra": payload.get("extra") or {},
                            "labels_status": "unlabelled",
                        },
                    }
                ],
            }
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an ATIF trajectory to DriftShield validation JSON")
    parser.add_argument("input", type=Path, help="ATIF trajectory_atif.json")
    parser.add_argument("--output", type=Path, required=True, help="Output DriftShield validation JSON")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    converted = convert_trajectory(payload, args.input.name)
    args.output.write_text(json.dumps(converted, indent=2) + "\\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
