"""DriftMonitor — the core wrapper that ties detection, storage, and alerting together."""

from __future__ import annotations

import logging
import time
import uuid
from functools import wraps
from typing import Any, Callable

from driftshield.alerts import AlertDispatcher
from driftshield.baseline import Calibrator
from driftshield.detectors import ActionLoopDetector, GoalDriftDetector, ResourceSpikeDetector
from driftshield.models import BaselineStats, DriftEvent, TraceEvent
from driftshield.storage import TraceStore

logger = logging.getLogger(__name__)


class DriftMonitor:
    """
    In-process wrapper for agentic AI systems.

    Decorates an agent's execution, intercepts events, and analyses them
    asynchronously. No separate server, no network hop, no infrastructure.

    Usage:
        monitor = DriftMonitor(
            agent_id="logistics-v2",
            alert_webhook="https://hooks.slack.com/...",
            calibration_runs=30,
        )
        agent = monitor.wrap(existing_agent)
        result = agent.invoke({"input": "optimise route for order #4821"})
    """

    def __init__(
        self,
        agent_id: str,
        alert_webhook: str | None = None,
        goal_description: str = "",
        calibration_runs: int = 30,
        db_path: str | None = None,
        # Detector config
        loop_window: int = 20,
        loop_max_repeats: int = 4,
        similarity_threshold: float = 0.5,
        spike_multiplier: float = 2.5,
        # Alert config
        min_alert_severity: str = "MED",
        alert_cooldown: float = 60.0,
    ):
        self.agent_id = agent_id
        self._current_run_id: str | None = None

        # Storage
        self.store = TraceStore(db_path=db_path)

        # Calibration
        self.calibrator = Calibrator(self.store, required_runs=calibration_runs)
        self._baseline: BaselineStats | None = self.store.get_baseline(agent_id)

        # Detectors
        self.action_loop = ActionLoopDetector(
            self.store, window_size=loop_window, max_repeats=loop_max_repeats
        )
        self.goal_drift = GoalDriftDetector(
            self.store,
            goal_description=goal_description,
            similarity_threshold=similarity_threshold,
        )
        self.resource_spike = ResourceSpikeDetector(
            self.store, spike_multiplier=spike_multiplier
        )
        self._detectors = [self.action_loop, self.goal_drift, self.resource_spike]

        # Alerts
        self.alerter = AlertDispatcher(
            webhook_url=alert_webhook,
            min_severity=min_alert_severity,
            cooldown_seconds=alert_cooldown,
        )

        # Callbacks
        self._on_drift_callbacks: list[Callable[[DriftEvent], None]] = []

        logger.info(
            f"DriftMonitor initialised for '{agent_id}' "
            f"(baseline: {'calibrated' if self._baseline and self._baseline.is_calibrated else 'pending'})"
        )

    # ── Public API ────────────────────────────────────────────────

    def wrap(self, agent: Any) -> Any:
        """
        Wrap a LangChain agent/chain to intercept execution.
        Returns a wrapped object with the same interface.
        """
        return _LangChainWrapper(self, agent)

    def start_run(self, run_id: str | None = None, goal: str | None = None) -> str:
        """Manually start a monitored run. Returns the run ID."""
        self._current_run_id = run_id or uuid.uuid4().hex[:12]
        if goal:
            self.goal_drift.set_goal(goal)
        logger.debug(f"Run started: {self._current_run_id}")
        return self._current_run_id

    def end_run(self, run_id: str | None = None) -> None:
        """End a run and update the baseline."""
        rid = run_id or self._current_run_id
        if rid:
            self._baseline = self.calibrator.update_baseline(self.agent_id)
        self._current_run_id = None

    def record_event(
        self,
        action_type: str,
        action_name: str,
        run_id: str | None = None,
        token_count: int = 0,
        input_data: dict | None = None,
        output_data: dict | None = None,
        duration_ms: float = 0.0,
        metadata: dict | None = None,
    ) -> list[DriftEvent]:
        """
        Record a trace event and run it through all detectors.
        Returns any drift events that were detected.
        """
        rid = run_id or self._current_run_id or uuid.uuid4().hex[:12]

        event = TraceEvent(
            agent_id=self.agent_id,
            run_id=rid,
            action_type=action_type,
            action_name=action_name,
            token_count=token_count,
            input_data=input_data or {},
            output_data=output_data or {},
            duration_ms=duration_ms,
            metadata=metadata or {},
        )

        # Store the trace
        self.store.save_trace(event)

        # Run all detectors
        drift_events = []
        for detector in self._detectors:
            try:
                drift = detector.check(event, self._baseline)
                if drift:
                    self.store.save_drift(drift)
                    drift_events.append(drift)

                    # Fire alert
                    self.alerter.send_sync(drift)

                    # Fire callbacks
                    for cb in self._on_drift_callbacks:
                        try:
                            cb(drift)
                        except Exception as e:
                            logger.warning(f"Drift callback error: {e}")

                    logger.warning(
                        f"DRIFT [{drift.severity.value}] {drift.detector.value}: {drift.message}"
                    )
            except Exception as e:
                logger.error(f"Detector '{detector.name()}' failed: {e}")

        return drift_events

    def on_drift(self, callback: Callable[[DriftEvent], None]) -> None:
        """Register a callback to be fired when drift is detected."""
        self._on_drift_callbacks.append(callback)

    def get_baseline(self) -> BaselineStats | None:
        """Get the current baseline stats."""
        return self._baseline

    def get_recent_alerts(self, hours: float = 24, limit: int = 50) -> list[DriftEvent]:
        """Get recent drift events."""
        since = time.time() - (hours * 3600)
        return self.store.get_drift_events(agent_id=self.agent_id, since=since, limit=limit)


class _LangChainWrapper:
    """Transparent wrapper around a LangChain agent/chain."""

    def __init__(self, monitor: DriftMonitor, agent: Any):
        self._monitor = monitor
        self._agent = agent

    def invoke(self, input_data: dict[str, Any], **kwargs: Any) -> Any:
        """Wrap the standard LangChain invoke method."""
        run_id = self._monitor.start_run(
            goal=input_data.get("input", "") or input_data.get("query", "")
        )

        start = time.time()
        try:
            # Record the invocation
            self._monitor.record_event(
                action_type="llm_request",
                action_name="agent_invoke",
                run_id=run_id,
                input_data=input_data,
            )

            # Call the real agent
            result = self._agent.invoke(input_data, **kwargs)

            elapsed_ms = (time.time() - start) * 1000
            output_text = ""
            if isinstance(result, dict):
                output_text = result.get("output", str(result))
            elif isinstance(result, str):
                output_text = result

            # Record completion
            self._monitor.record_event(
                action_type="llm_request",
                action_name="agent_complete",
                run_id=run_id,
                output_data={"text": output_text},
                duration_ms=elapsed_ms,
                token_count=self._estimate_tokens(output_text),
            )

            return result

        except Exception as e:
            self._monitor.record_event(
                action_type="state_transition",
                action_name="agent_error",
                run_id=run_id,
                output_data={"error": str(e)},
                duration_ms=(time.time() - start) * 1000,
            )
            raise
        finally:
            self._monitor.end_run(run_id)

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the wrapped agent."""
        return getattr(self._agent, name)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate (4 chars per token)."""
        return len(text) // 4 if text else 0
