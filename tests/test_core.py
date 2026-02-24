"""Tests for DriftShield core functionality."""

import os
import tempfile
import time

import pytest

from driftshield import DriftMonitor, TraceEvent
from driftshield.detectors import ActionLoopDetector, ResourceSpikeDetector
from driftshield.models import BaselineStats, DetectorType, Severity
from driftshield.storage import TraceStore


@pytest.fixture
def tmp_db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture
def store(tmp_db):
    return TraceStore(db_path=tmp_db)


@pytest.fixture
def monitor(tmp_db):
    return DriftMonitor(
        agent_id="test-agent",
        db_path=tmp_db,
        calibration_runs=5,
        loop_max_repeats=3,
    )


# ── Storage Tests ─────────────────────────────────────────────


class TestTraceStore:
    def test_save_and_retrieve_trace(self, store):
        event = TraceEvent(
            agent_id="agent-1",
            run_id="run-1",
            action_type="tool_call",
            action_name="search_db",
            token_count=100,
        )
        store.save_trace(event)

        traces = store.get_traces(agent_id="agent-1")
        assert len(traces) == 1
        assert traces[0].action_name == "search_db"
        assert traces[0].token_count == 100

    def test_get_recent_actions(self, store):
        for i, name in enumerate(["search", "format", "search", "format", "search"]):
            event = TraceEvent(
                agent_id="agent-1",
                run_id="run-1",
                action_type="tool_call",
                action_name=name,
                timestamp=time.time() + i,
            )
            store.save_trace(event)

        actions = store.get_recent_actions("agent-1", "run-1", window=10)
        assert actions == ["search", "format", "search", "format", "search"]

    def test_run_stats(self, store):
        for i in range(5):
            store.save_trace(TraceEvent(
                agent_id="agent-1",
                run_id="run-1",
                action_type="tool_call",
                action_name=f"tool_{i}",
                token_count=100,
                duration_ms=50.0,
            ))

        stats = store.get_run_stats("agent-1", "run-1")
        assert stats["event_count"] == 5
        assert stats["total_tokens"] == 500
        assert stats["tool_calls"] == 5


# ── Action Loop Detector Tests ────────────────────────────────


class TestActionLoopDetector:
    def test_detects_single_tool_loop(self, store):
        detector = ActionLoopDetector(store, max_repeats=3)

        run_id = "run-loop"
        for i in range(5):
            store.save_trace(TraceEvent(
                agent_id="agent-1",
                run_id=run_id,
                action_type="tool_call",
                action_name="search_db",
                timestamp=time.time() + i,
            ))

        event = TraceEvent(
            agent_id="agent-1",
            run_id=run_id,
            action_type="tool_call",
            action_name="search_db",
        )

        drift = detector.check(event, None)
        assert drift is not None
        assert drift.detector == DetectorType.ACTION_LOOP
        assert "search_db" in drift.message

    def test_no_false_positive_on_variety(self, store):
        detector = ActionLoopDetector(store, max_repeats=3)

        run_id = "run-ok"
        for i, name in enumerate(["search", "format", "save", "notify", "complete"]):
            store.save_trace(TraceEvent(
                agent_id="agent-1",
                run_id=run_id,
                action_type="tool_call",
                action_name=name,
                timestamp=time.time() + i,
            ))

        event = TraceEvent(
            agent_id="agent-1",
            run_id=run_id,
            action_type="tool_call",
            action_name="complete",
        )

        drift = detector.check(event, None)
        assert drift is None

    def test_ignores_non_tool_calls(self, store):
        detector = ActionLoopDetector(store, max_repeats=3)

        event = TraceEvent(
            agent_id="agent-1",
            run_id="run-1",
            action_type="llm_request",
            action_name="generate",
        )

        drift = detector.check(event, None)
        assert drift is None


# ── Resource Spike Detector Tests ─────────────────────────────


class TestResourceSpikeDetector:
    def test_detects_absolute_token_spike(self, store):
        detector = ResourceSpikeDetector(store, absolute_token_limit=1000)

        run_id = "run-spike"
        events = []
        for i in range(20):
            event = TraceEvent(
                agent_id="agent-1",
                run_id=run_id,
                action_type="llm_request",
                action_name="generate",
                token_count=100,
            )
            drift_result = detector.check(event, None)
            events.append(drift_result)

        # Should trigger once we exceed 1000 tokens
        drifts = [d for d in events if d is not None]
        assert len(drifts) > 0
        assert drifts[0].detector == DetectorType.RESOURCE_SPIKE

    def test_detects_baseline_spike(self, store):
        detector = ResourceSpikeDetector(store, spike_multiplier=2.0)

        baseline = BaselineStats(
            agent_id="agent-1",
            mean_tokens_per_run=200,
            std_tokens_per_run=50,
            is_calibrated=True,
        )

        run_id = "run-baseline-spike"
        # Burn 500 tokens (well above 200 + 2*50 = 300)
        for i in range(5):
            event = TraceEvent(
                agent_id="agent-1",
                run_id=run_id,
                action_type="llm_request",
                action_name="generate",
                token_count=100,
            )
            drift = detector.check(event, baseline)

        assert drift is not None
        assert "token_burn" in drift.message


# ── Monitor Integration Tests ─────────────────────────────────


class TestDriftMonitor:
    def test_record_event_returns_drift(self, monitor):
        """Recording many identical tool calls should trigger loop detection."""
        run_id = monitor.start_run()

        drifts = []
        for i in range(10):
            result = monitor.record_event(
                action_type="tool_call",
                action_name="search_db",
                run_id=run_id,
            )
            drifts.extend(result)

        assert len(drifts) > 0
        assert any(d.detector == DetectorType.ACTION_LOOP for d in drifts)

    def test_clean_run_no_drift(self, monitor):
        """A normal run with varied actions should not trigger drift."""
        run_id = monitor.start_run()

        all_drifts = []
        for name in ["search", "format", "save"]:
            drifts = monitor.record_event(
                action_type="tool_call",
                action_name=name,
                run_id=run_id,
                token_count=50,
            )
            all_drifts.extend(drifts)

        monitor.end_run(run_id)
        assert len(all_drifts) == 0

    def test_baseline_builds_after_enough_runs(self, monitor):
        """Baseline should be calibrated after required_runs."""
        for run_num in range(6):
            run_id = monitor.start_run()
            for name in ["fetch", "process", "store"]:
                monitor.record_event(
                    action_type="tool_call",
                    action_name=name,
                    run_id=run_id,
                    token_count=100,
                )
            monitor.end_run(run_id)

        baseline = monitor.get_baseline()
        assert baseline is not None
        assert baseline.is_calibrated
        assert baseline.mean_tokens_per_run > 0

    def test_on_drift_callback(self, monitor):
        """Drift callbacks should fire."""
        fired = []
        monitor.on_drift(lambda d: fired.append(d))

        run_id = monitor.start_run()
        for i in range(10):
            monitor.record_event(
                action_type="tool_call",
                action_name="stuck_tool",
                run_id=run_id,
            )

        assert len(fired) > 0


# ── Model Tests ───────────────────────────────────────────────


class TestModels:
    def test_severity_from_score(self):
        assert Severity.from_score(0.3) == Severity.LOW
        assert Severity.from_score(0.5) == Severity.MEDIUM
        assert Severity.from_score(0.7) == Severity.HIGH
        assert Severity.from_score(0.95) == Severity.CRITICAL

    def test_trace_event_roundtrip(self):
        event = TraceEvent(
            agent_id="a", run_id="r", action_type="tool_call", action_name="test"
        )
        d = event.to_dict()
        restored = TraceEvent.from_dict(d)
        assert restored.agent_id == "a"
        assert restored.action_name == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
