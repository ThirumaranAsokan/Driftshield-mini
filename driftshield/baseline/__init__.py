"""Baseline calibrator — builds statistical norms from initial agent runs."""

from __future__ import annotations

import logging

import numpy as np

from driftshield.models import BaselineStats
from driftshield.storage import TraceStore

logger = logging.getLogger(__name__)


class Calibrator:
    """
    During calibration (first N runs), records statistical norms:
    mean tools per run, typical token burn, common action sequences,
    and goal-alignment embeddings.
    """

    def __init__(self, store: TraceStore, required_runs: int = 30):
        self.store = store
        self.required_runs = required_runs

    def update_baseline(self, agent_id: str) -> BaselineStats:
        """
        Recalculate baseline from all stored runs for this agent.
        Returns updated BaselineStats (also persisted to store).
        """
        run_ids = self.store.get_run_ids(agent_id, limit=self.required_runs)

        if not run_ids:
            baseline = BaselineStats(agent_id=agent_id)
            self.store.save_baseline(baseline)
            return baseline

        tokens_per_run = []
        tools_per_run = []
        durations = []
        all_sequences = []

        for run_id in run_ids:
            stats = self.store.get_run_stats(agent_id, run_id)
            tokens_per_run.append(stats["total_tokens"])
            tools_per_run.append(stats["tool_calls"])
            durations.append(stats["total_duration_ms"])

            # Collect action sequences
            actions = self.store.get_recent_actions(agent_id, run_id, window=50)
            if actions:
                all_sequences.append(actions)

        tokens = np.array(tokens_per_run, dtype=np.float64)
        tools = np.array(tools_per_run, dtype=np.float64)
        durs = np.array(durations, dtype=np.float64)

        is_calibrated = len(run_ids) >= self.required_runs

        baseline = BaselineStats(
            agent_id=agent_id,
            calibration_runs=len(run_ids),
            mean_tokens_per_run=float(np.mean(tokens)) if len(tokens) > 0 else 0.0,
            std_tokens_per_run=float(np.std(tokens)) if len(tokens) > 1 else 0.0,
            mean_tools_per_run=float(np.mean(tools)) if len(tools) > 0 else 0.0,
            std_tools_per_run=float(np.std(tools)) if len(tools) > 1 else 0.0,
            mean_duration_ms=float(np.mean(durs)) if len(durs) > 0 else 0.0,
            std_duration_ms=float(np.std(durs)) if len(durs) > 1 else 0.0,
            common_sequences=self._find_common_sequences(all_sequences),
            is_calibrated=is_calibrated,
        )

        self.store.save_baseline(baseline)

        if is_calibrated:
            logger.info(
                f"Baseline calibrated for '{agent_id}' "
                f"({baseline.calibration_runs} runs, "
                f"μ_tokens={baseline.mean_tokens_per_run:.0f}, "
                f"μ_tools={baseline.mean_tools_per_run:.1f})"
            )
        else:
            logger.info(
                f"Baseline partial for '{agent_id}' "
                f"({baseline.calibration_runs}/{self.required_runs} runs)"
            )

        return baseline

    def _find_common_sequences(
        self, all_sequences: list[list[str]], min_length: int = 2, top_n: int = 5
    ) -> list[list[str]]:
        """Extract the most common action subsequences across runs."""
        from collections import Counter

        subseq_counts: Counter[tuple[str, ...]] = Counter()

        for seq in all_sequences:
            seen: set[tuple[str, ...]] = set()
            for length in range(min_length, min(len(seq) + 1, 5)):
                for i in range(len(seq) - length + 1):
                    subseq = tuple(seq[i : i + length])
                    if subseq not in seen:
                        subseq_counts[subseq] += 1
                        seen.add(subseq)

        return [list(seq) for seq, _ in subseq_counts.most_common(top_n)]
