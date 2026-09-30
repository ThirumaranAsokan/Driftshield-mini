"""External analysis of Nebius SWE-agent trajectories."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

TOOL_NAMES = {"apply_patch", "bash", "cat", "cd", "find", "git", "grep", "ls",
              "mkdir", "python", "pytest", "rg", "sed", "tail", "touch", "vim"}

@dataclass(frozen=True)
class TrajectorySignals:
    steps: int
    action_names: tuple[str, ...]
    repeated_action: bool
    max_consecutive_action: int
    token_estimate: int

def _trajectory_items(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]

def _normalise_action(text: str) -> str | None:
    value = " ".join(text.strip().split())
    if not value:
        return None
    first = value.split()[0].lower()
    if first not in TOOL_NAMES:
        return None
    return value[:500]

def extract_actions(text: str) -> list[str]:
    actions = []
    for line in text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered.startswith("action:") or lowered.startswith("command:"):
            action = _normalise_action(stripped.split(":", 1)[1])
            if action:
                actions.append(action)
    marker = "will execute following command"
    if marker in text.lower():
        chunks = text.split("```")
        for chunk in chunks[1::2]:
            action = _normalise_action(chunk.strip().splitlines()[0])
            if action:
                actions.append(action)
    return actions

def _max_consecutive(values: Iterable[str]) -> int:
    best = current = 0
    previous = None
    for value in values:
        if value == previous:
            current += 1
        else:
            current = 1
            previous = value
        best = max(best, current)
    return best

def extract_signals(row: dict[str, Any]) -> TrajectorySignals:
    items = _trajectory_items(row.get("trajectory", []))
    ai_texts = tuple(str(item.get("text", "")) for item in items
                     if str(item.get("role", "")).lower() == "ai")
    actions = tuple(action for text in ai_texts for action in extract_actions(text))
    token_estimate = sum(max(1, len(text) // 4) for text in ai_texts)
    return TrajectorySignals(len(items), actions, len(set(actions)) < len(actions) if actions else False,
                             _max_consecutive(actions), token_estimate)

def analyse_rows(rows: Iterable[dict[str, Any]], loop_repeats: int = 4,
                 token_limit: int = 50000) -> dict[str, Any]:
    groups = {"target_true": Counter(), "target_false": Counter()}
    counts = Counter()
    total = 0
    extracted = 0
    for row in rows:
        total += 1
        group = "target_true" if bool(row.get("target", False)) else "target_false"
        signals = extract_signals(row)
        loop = signals.max_consecutive_action >= loop_repeats
        resource = signals.token_estimate > token_limit
        groups[group]["trajectories"] += 1
        groups[group]["loop_signal"] += int(loop)
        groups[group]["resource_signal"] += int(resource)
        groups[group]["action_extracted"] += int(bool(signals.action_names))
        extracted += int(bool(signals.action_names))
        counts["loop_signal"] += int(loop)
        counts["resource_signal"] += int(resource)
        counts["estimated_tokens"] += signals.token_estimate
    return {"dataset": "nebius/SWE-agent-trajectories", "rows_analyzed": total,
            "action_extraction_rate": round(extracted / total, 4) if total else 0.0,
            "overall": dict(counts),
            "by_target": {key: dict(value) for key, value in groups.items()},
            "accuracy_metrics": "not computed: dataset has no DriftShield detector labels"}

def load_rows(limit: int | None, split: str) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install the optional external-validation dependency first.") from exc
    dataset = load_dataset("nebius/SWE-agent-trajectories", split=split, streaming=True)
    if limit is None:
        return dataset
    return (row for index, row in enumerate(dataset) if index < limit)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--loop-repeats", type=int, default=4)
    parser.add_argument("--token-limit", type=int, default=50000)
    args = parser.parse_args()
    result = analyse_rows(load_rows(args.limit, args.split), args.loop_repeats, args.token_limit)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "
", encoding="utf-8")
    else:
        print(rendered)

if __name__ == "__main__":
    main()