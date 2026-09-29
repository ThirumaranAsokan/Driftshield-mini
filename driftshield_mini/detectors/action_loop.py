"""Action Loop Detector — catches agents stuck in repetitive tool-call cycles."""

from __future__ import annotations

from collections import Counter

from driftshield.detectors.base import BaseDetector
from driftshield.models import (
    BaselineStats,
    DetectorType,
    DriftEvent,
    Severity,
    TraceEvent,
)
from driftshield.storage import TraceStore


class ActionLoopDetector(BaseDetector):
    """
    Monitors a sliding window of recent tool calls and flags when the same
    tool (or tool sequence) fires more than `max_repeats` times without
    meaningful state change.

    Uses simple sequence matching — no ML required.
    """

    def __init__(
        self,
        store: TraceStore,
        window_size: int = 20,
        max_repeats: int = 4,
        sequence_length: int = 3,
        enabled: bool = True,
    ):
        super().__init__(store, enabled)
        self.window_size = window_size
        self.max_repeats = max_repeats
        self.sequence_length = sequence_length

    def name(self) -> str:
        return "action_loop"

    def check(
        self,
        event: TraceEvent,
        baseline: BaselineStats | None,
    ) -> DriftEvent | None:
        if not self.enabled or event.action_type != "tool_call":
            return None

        recent = self.store.get_recent_actions(
            event.agent_id, event.run_id, window=self.window_size
        )

        if len(recent) < self.max_repeats:
            return None

        # Check 1: Single tool repeated too many times
        drift = self._check_single_repeat(recent, event)
        if drift:
            return drift

        # Check 2: Repeating sequence (e.g., A→B→C→A→B→C)
        drift = self._check_sequence_repeat(recent, event)
        if drift:
            return drift

        return None

    def _check_single_repeat(
        self, recent: list[str], event: TraceEvent
    ) -> DriftEvent | None:
        """Detect a single tool called repeatedly."""
        if len(recent) < self.max_repeats:
            return None

        tail = recent[-self.max_repeats :]
        if len(set(tail)) == 1:
            tool_name = tail[0]
            repeat_count = 0
            for action in reversed(recent):
                if action == tool_name:
                    repeat_count += 1
                else:
                    break

            score = min(1.0, repeat_count / (self.max_repeats * 2))
            return DriftEvent(
                agent_id=event.agent_id,
                run_id=event.run_id,
                detector=DetectorType.ACTION_LOOP,
                severity=Severity.from_score(score),
                score=score,
                message=f"Action loop: {tool_name} called {repeat_count}x consecutively",
                suggested_action=f"Check {tool_name} input/output for stale data or error loops",
                context={
                    "tool_name": tool_name,
                    "repeat_count": repeat_count,
                    "recent_actions": recent[-10:],
                },
            )
        return None

    def _check_sequence_repeat(
        self, recent: list[str], event: TraceEvent
    ) -> DriftEvent | None:
        """Detect repeating sequences like A→B→A→B or A→B→C→A→B→C."""
        for seq_len in range(2, self.sequence_length + 1):
            if len(recent) < seq_len * self.max_repeats:
                continue

            # Extract possible repeating pattern from end of list
            pattern = recent[-seq_len:]
            repeat_count = 0
            idx = len(recent) - seq_len

            while idx >= 0:
                window = recent[idx : idx + seq_len]
                if window == pattern:
                    repeat_count += 1
                    idx -= seq_len
                else:
                    break

            if repeat_count >= self.max_repeats:
                seq_str = " → ".join(pattern)
                score = min(1.0, repeat_count / (self.max_repeats * 2))
                return DriftEvent(
                    agent_id=event.agent_id,
                    run_id=event.run_id,
                    detector=DetectorType.ACTION_LOOP,
                    severity=Severity.from_score(score),
                    score=score,
                    message=f"Action loop: sequence [{seq_str}] repeated {repeat_count}x",
                    suggested_action="Review agent logic for circular tool dependencies",
                    context={
                        "sequence": pattern,
                        "repeat_count": repeat_count,
                        "recent_actions": recent[-15:],
                    },
                )
        return None
