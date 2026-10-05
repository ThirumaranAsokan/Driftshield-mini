"""External analysis of published tau2-bench (tau3-bench) trajectories.

This module deliberately reports behavioural associations, not DriftShield
detector accuracy. tau2-bench reward/termination fields are benchmark outcomes,
not independent labels for action_loop, goal_drift, or resource_spike.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from statistics import quantiles
from typing import Any


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _messages(sim: dict[str, Any]) -> list[dict[str, Any]]:
    return [m for m in _as_list(sim.get("messages")) if isinstance(m, dict)]


def _tool_names(sim: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for message in _messages(sim):
        for call in _as_list(message.get("tool_calls")):
            if isinstance(call, dict) and isinstance(call.get("name"), str):
                names.append(call["name"])
    return names


def _max_consecutive(values: Iterable[str]) -> int:
    best = current = 0
    previous: str | None = None
    for value in values:
        if value == previous:
            current += 1
        else:
            current = 1
            previous = value
        best = max(best, current)
    return best


def _usage_tokens(sim: dict[str, Any]) -> int | None:
    total = 0
    found = False
    for message in _messages(sim):
        usage = message.get("usage")
        if not isinstance(usage, dict):
            continue
        value = usage.get("total_tokens")
        if isinstance(value, (int, float)):
            total += int(value)
            found = True
            continue
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        message_total = 0
        if isinstance(input_tokens, (int, float)):
            message_total += int(input_tokens)
        if isinstance(output_tokens, (int, float)):
            message_total += int(output_tokens)
        if message_total:
            total += message_total
            found = True
    return total if found else None


def _agent_cost(sim: dict[str, Any]) -> float | None:
    value = sim.get("agent_cost")
    return float(value) if isinstance(value, (int, float)) else None


def _reward(sim: dict[str, Any]) -> float | None:
    reward_info = sim.get("reward_info")
    if isinstance(reward_info, dict) and isinstance(reward_info.get("reward"), (int, float)):
        return float(reward_info["reward"])
    return None


def _failure(sim: dict[str, Any]) -> bool | None:
    reward = _reward(sim)
    if reward is None:
        return None
    return reward < 1.0


def _percentile_threshold(values: list[float], percentile: float = 0.99) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    qs = quantiles(values, n=100, method="inclusive")
    return qs[int(percentile * 100) - 1]


def _confusion(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    # Outcome association only: positive outcome means benchmark failure.
    tp = sum(signal and failed for signal, failed in pairs)
    fp = sum(signal and not failed for signal, failed in pairs)
    fn = sum(not signal and failed for signal, failed in pairs)
    tn = sum(not signal and not failed for signal, failed in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "fpr": round(fpr, 6),
    }


def analyse_files(paths: list[Path], loop_repeats: int = 4) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        simulations = _as_list(payload.get("simulations"))
        files.append({
            "file": path.name,
            "timestamp": payload.get("timestamp"),
            "git_commit": (payload.get("info") or {}).get("git_commit")
            if isinstance(payload.get("info"), dict) else None,
            "simulations": len(simulations),
        })
        rows.extend(sim for sim in simulations if isinstance(sim, dict))

    telemetry_cost = []
    telemetry_tokens = []
    records = []
    for sim in rows:
        tools = _tool_names(sim)
        max_repeat = _max_consecutive(tools)
        cost = _agent_cost(sim)
        tokens = _usage_tokens(sim)
        if cost is not None:
            telemetry_cost.append(cost)
        if tokens is not None:
            telemetry_tokens.append(float(tokens))
        records.append({
            "failure": _failure(sim),
            "max_repeat": max_repeat,
            "termination": sim.get("termination_reason"),
            "duration": sim.get("duration"),
            "cost": cost,
            "tokens": tokens,
            "tool_calls": len(tools),
            "review": sim.get("review"),
        })

    cost_threshold = _percentile_threshold(telemetry_cost)
    token_threshold = _percentile_threshold(telemetry_tokens)
    loop_pairs = [
        (r["max_repeat"] >= loop_repeats, r["failure"])
        for r in records if r["failure"] is not None
    ]
    cost_pairs = [
        (r["cost"] is not None and cost_threshold is not None and r["cost"] >= cost_threshold, r["failure"])
        for r in records if r["failure"] is not None and r["cost"] is not None
    ]
    token_pairs = [
        (r["tokens"] is not None and token_threshold is not None and r["tokens"] >= token_threshold, r["failure"])
        for r in records if r["failure"] is not None and r["tokens"] is not None
    ]

    termination_counts = Counter(str(r["termination"]) for r in records if r["termination"])
    review_error_counts = Counter()
    review_present = 0
    for r in records:
        review = r["review"]
        if not isinstance(review, dict):
            continue
        review_present += 1
        for error in _as_list(review.get("errors")):
            if isinstance(error, dict):
                error_type = error.get("error_type")
                source = error.get("source")
                if error_type:
                    review_error_counts[f"{source}:{error_type}"] += 1

    return {
        "dataset": "sierra-research/tau2-bench",
        "source": "published data/tau2/results/final/*.json",
        "files": files,
        "rows_analyzed": len(records),
        "outcome_definition": "benchmark failure means reward < 1.0; this is NOT a DriftShield detector label",
        "action_loop_association": {
            "loop_definition": f"four or more consecutive identical tool names (threshold={loop_repeats})",
            "metrics": _confusion(loop_pairs),
        },
        "resource_association": {
            "cost": {
                "available_rows": len(telemetry_cost),
                "p99_threshold_usd": round(cost_threshold, 8) if cost_threshold is not None else None,
                "metrics": _confusion(cost_pairs),
            },
            "provider_token_usage": {
                "available_rows": len(telemetry_tokens),
                "p99_threshold_tokens": int(token_threshold) if token_threshold is not None else None,
                "metrics": _confusion(token_pairs),
            },
        },
        "termination_reasons": dict(termination_counts),
        "review": {
            "rows_with_review": review_present,
            "error_types": dict(review_error_counts),
        },
        "goal_drift": {
            "status": "not_directly_evaluable",
            "reason": "tau2-bench does not provide an independent goal_drift ground-truth label; task reward and agent-error reviews must not be relabelled as goal drift",
        },
        "accuracy_warning": "These are external benchmark outcome associations, NOT DriftShield precision/recall/F1/FPR.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--loop-repeats", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyse_files(args.files, args.loop_repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
