"""Resource Spike Detector — flags abnormal consumption patterns."""

from __future__ import annotations

import time

from driftshield.detectors.base import BaseDetector
from driftshield.models import (
    BaselineStats,
    DetectorType,
    DriftEvent,
    Severity,
    TraceEvent,
)
from driftshield.storage import TraceStore


class ResourceSpikeDetector(BaseDetector):
    """
    Flags abnormal consumption patterns: token burn rate, execution time,
    or API call frequency exceeding baseline norms by a configurable
    multiplier (default: 2.5× standard deviation).
    """

    def __init__(
        self,
        store: TraceStore,
        spike_multiplier: float = 2.5,
        absolute_token_limit: int = 50_000,
        absolute_duration_limit_ms: float = 300_000,  # 5 minutes
        enabled: bool = True,
    ):
        super().__init__(store, enabled)
        self.spike_multiplier = spike_multiplier
        self.absolute_token_limit = absolute_token_limit
        self.absolute_duration_limit_ms = absolute_duration_limit_ms

        # Running counters per run (reset on new run)
        self._run_counters: dict[str, dict] = {}

    def name(self) -> str:
        return "resource_spike"

    def _get_run_counter(self, run_id: str) -> dict:
        if run_id not in self._run_counters:
            self._run_counters[run_id] = {
                "total_tokens": 0,
                "total_duration_ms": 0.0,
                "tool_calls": 0,
                "llm_calls": 0,
                "start_time": time.time(),
            }
            # Cleanup old counters (keep last 10)
            if len(self._run_counters) > 10:
                oldest = sorted(self._run_counters.keys())[0]
                del self._run_counters[oldest]
        return self._run_counters[run_id]

    def check(
        self,
        event: TraceEvent,
        baseline: BaselineStats | None,
    ) -> DriftEvent | None:
        if not self.enabled:
            return None

        counter = self._get_run_counter(event.run_id)
        counter["total_tokens"] += event.token_count
        counter["total_duration_ms"] += event.duration_ms
        if event.action_type == "tool_call":
            counter["tool_calls"] += 1
        elif event.action_type == "llm_request":
            counter["llm_calls"] += 1

        # Check against baseline (statistical detection)
        if baseline and baseline.is_calibrated:
            drift = self._check_baseline_spike(event, baseline, counter)
            if drift:
                return drift

        # Check against absolute limits (safety net even without baseline)
        drift = self._check_absolute_limits(event, counter)
        if drift:
            return drift

        return None

    def _check_baseline_spike(
        self,
        event: TraceEvent,
        baseline: BaselineStats,
        counter: dict,
    ) -> DriftEvent | None:
        """Check if current run metrics exceed baseline by spike_multiplier × std."""

        checks = [
            (
                "token_burn",
                counter["total_tokens"],
                baseline.mean_tokens_per_run,
                baseline.std_tokens_per_run,
                "tokens",
            ),
            (
                "duration",
                counter["total_duration_ms"],
                baseline.mean_duration_ms,
                baseline.std_duration_ms,
                "ms",
            ),
            (
                "tool_calls",
                counter["tool_calls"],
                baseline.mean_tools_per_run,
                baseline.std_tools_per_run,
                "calls",
            ),
        ]

        for metric_name, current, mean, std, unit in checks:
            if mean == 0 and std == 0:
                continue

            threshold = mean + self.spike_multiplier * max(std, mean * 0.1)
            if current > threshold and current > mean * 1.5:
                score = min(1.0, (current - threshold) / max(threshold, 1))
                return DriftEvent(
                    agent_id=event.agent_id,
                    run_id=event.run_id,
                    detector=DetectorType.RESOURCE_SPIKE,
                    severity=Severity.from_score(score),
                    score=score,
                    message=(
                        f"Resource spike: {metric_name} at {current:.0f} {unit} "
                        f"(baseline: {mean:.0f} ± {std:.0f})"
                    ),
                    suggested_action=(
                        f"Check for malformed input or error loops causing elevated {metric_name}"
                    ),
                    context={
                        "metric": metric_name,
                        "current": current,
                        "baseline_mean": mean,
                        "baseline_std": std,
                        "threshold": threshold,
                        "run_totals": dict(counter),
                    },
                )
        return None

    def _check_absolute_limits(
        self,
        event: TraceEvent,
        counter: dict,
    ) -> DriftEvent | None:
        """Safety-net checks even without a calibrated baseline."""

        if counter["total_tokens"] > self.absolute_token_limit:
            score = min(
                1.0,
                (counter["total_tokens"] - self.absolute_token_limit)
                / self.absolute_token_limit,
            )
            return DriftEvent(
                agent_id=event.agent_id,
                run_id=event.run_id,
                detector=DetectorType.RESOURCE_SPIKE,
                severity=Severity.from_score(max(score, 0.7)),
                score=max(score, 0.7),
                message=(
                    f"Resource spike: token count {counter['total_tokens']:,} "
                    f"exceeds absolute limit ({self.absolute_token_limit:,})"
                ),
                suggested_action="Investigate agent run — token consumption is abnormally high",
                context={
                    "metric": "absolute_token_limit",
                    "current_tokens": counter["total_tokens"],
                    "limit": self.absolute_token_limit,
                },
            )

        elapsed = (time.time() - counter["start_time"]) * 1000
        if elapsed > self.absolute_duration_limit_ms:
            return DriftEvent(
                agent_id=event.agent_id,
                run_id=event.run_id,
                detector=DetectorType.RESOURCE_SPIKE,
                severity=Severity.HIGH,
                score=0.8,
                message=(
                    f"Resource spike: run duration {elapsed / 1000:.1f}s "
                    f"exceeds limit ({self.absolute_duration_limit_ms / 1000:.0f}s)"
                ),
                suggested_action="Agent may be hung or stuck — consider terminating the run",
                context={
                    "metric": "absolute_duration_limit",
                    "elapsed_ms": elapsed,
                    "limit_ms": self.absolute_duration_limit_ms,
                },
            )

        return None
