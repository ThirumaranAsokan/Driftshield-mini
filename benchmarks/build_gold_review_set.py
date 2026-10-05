"""Build a deterministic blinded review set from nebius/SWE-agent-trajectories.

This script NEVER derives DriftShield labels from the dataset's target/exit_status.
Those fields are retained only in a private manifest for later audit and are omitted
from the reviewer file. Reviewers label action_loop, goal_drift, and resource_spike
independently of DriftShield output.

The sample uses deterministic reservoir sampling over the full streaming training
split so the review set is not biased toward the first repositories/tasks.

Usage:
  python benchmarks/build_gold_review_set.py --limit 300 --output validation/gold_review.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def stable_id(row: dict[str, Any]) -> str:
    raw = "|".join(
        str(row.get(key, ""))
        for key in ("instance_id", "model_name", "trajectory")
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def load_rows(limit: int, seed: int) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            'Install external validation first: python -m pip install -e ".[external-validation]"'
        ) from exc

    if limit <= 0:
        raise ValueError("limit must be positive")

    dataset = load_dataset(
        "nebius/SWE-agent-trajectories",
        split="train",
        streaming=True,
    )
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []

    for index, row in enumerate(dataset):
        item = dict(row)
        if index < limit:
            reservoir.append(item)
            continue

        slot = rng.randrange(index + 1)
        if slot < limit:
            reservoir[slot] = item

    return reservoir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.limit, args.seed)
    reviewer_rows = []
    private_manifest = []

    for index, row in enumerate(rows):
        review_id = stable_id(row)
        reviewer_rows.append(
            {
                "review_id": review_id,
                "source": "nebius/SWE-agent-trajectories",
                "source_index": index,
                "instance_id": str(row.get("instance_id", "")),
                "goal_or_task": (
                    "Review the complete trajectory below. Treat the first user/task "
                    "instruction in the trajectory as the task goal."
                ),
                "trajectory": row.get("trajectory", []),
                "labels": {
                    "action_loop": None,
                    "goal_drift": None,
                    "resource_spike": None,
                },
                "review_reason": {
                    "action_loop": "",
                    "goal_drift": "",
                    "resource_spike": "",
                },
                "reviewer_id": "",
                "review_confidence": "",
            }
        )
        private_manifest.append(
            {
                "review_id": review_id,
                "instance_id": row.get("instance_id"),
                "model_name": row.get("model_name"),
                "target": row.get("target"),
                "exit_status": row.get("exit_status"),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "nebius/SWE-agent-trajectories",
                "purpose": "independent DriftShield detector labelling",
                "label_status": "unlabelled",
                "sampling": {
                    "method": "deterministic reservoir sampling over full train split",
                    "seed": args.seed,
                    "sample_size": len(reviewer_rows),
                },
                "reviewer_instructions": "See docs/gold-validation.md",
                "runs": reviewer_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    private_path = args.output.with_name(args.output.stem + ".private-manifest.json")
    private_path.write_text(
        json.dumps(private_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Created {len(reviewer_rows)} blinded review records: {args.output}")
    print(f"Created private audit manifest: {private_path}")


if __name__ == "__main__":
    main()
