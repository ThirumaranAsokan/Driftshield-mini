"""Focused unit tests for the core detector decision boundaries."""

from __future__ import annotations

import pytest

from driftshield_mini.detectors.action_loop import ActionLoopDetector
from driftshield_mini.detectors.resource_spike import ResourceSpikeDetector
from driftshield_mini.models import BaselineStats, DetectorType, Severity, TraceEvent
from driftshield_mini.storage import TraceStore


@pytest.fixture()
def store(tmp_path):
    return TraceStore(tmp_path / "detectors.db")


def event(run_id: str, action_name: str, *, action_type: str = "tool_call",
          token_count: int = 0, duration_ms: float = 0.0) -> TraceEvent:
    return TraceEvent(
        agent_id="detector-tests",
        run_id=run_id,
        action_type=action_type,
        action_name=action_name,
        token_count=token_count,
        duration_ms=duration_ms,
    )


def save(store: TraceStore, trace: TraceEvent) -> None:
    store.save_trace(trace)


def test_action_loop_triggers_at_repeat_boundary(store):
    detector = ActionLoopDetector(store, window_size=20, max_repeats=4)
    traces = [event("run-1", "search") for _ in range(4)]
    for trace in traces:
        save(store, trace)

    drift = detector.check(traces[-1], None)

    assert drift is not None
    assert drift.detector is DetectorType.ACTION_LOOP
    assert drift.context["repeat_count"] == 4


def test_action_loop_does_not_trigger_before_repeat_boundary(store):
    detector = ActionLoopDetector(store, window_size=20, max_repeats=4)
    traces = [event("run-1", "search") for _ in range(3)]
    for trace in traces:
        save(store, trace)

    assert detector.check(traces[-1], None) is None


def test_action_loop_window_boundary_ignores_old_actions(store):
    detector = ActionLoopDetector(store, window_size=4, max_repeats=4)
    traces = [
        event("run-1", "other"),
        event("run-1", "search"),
        event("run-1", "search"),
        event("run-1", "search"),
        event("run-1", "search"),
    ]
    for trace in traces:
        save(store, trace)

    drift = detector.check(traces[-1], None)

    assert drift is not None
    assert drift.context["repeat_count"] == 4


def test_action_loop_does_not_cross_run_boundaries(store):
    detector = ActionLoopDetector(store, window_size=20, max_repeats=4)
    traces = [
        event("run-a", "search"),
        event("run-b", "search"),
        event("run-a", "search"),
        event("run-b", "search"),
        event("run-a", "search"),
        event("run-b", "search"),
        event("run-a", "search"),
    ]
    for trace in traces:
        save(store, trace)

    assert detector.check(traces[-1], None) is not None
    assert detector.check(traces[1], None) is None


def test_action_loop_sequence_boundary(store):
    detector = ActionLoopDetector(store, window_size=20, max_repeats=4, sequence_length=2)
    traces = [event("run-1", name) for name in ("search", "format") * 4]
    for trace in traces:
        save(store, trace)

    drift = detector.check(traces[-1], None)

    assert drift is not None
    assert drift.context["sequence"] == ["search", "format"]
    assert drift.context["repeat_count"] == 4


def test_action_loop_sequence_below_boundary(store):
    detector = ActionLoopDetector(store, window_size=20, max_repeats=4, sequence_length=2)
    traces = [event("run-1", name) for name in ("search", "format") * 3]
    for trace in traces:
        save(store, trace)

    assert detector.check(traces[-1], None) is None


@pytest.mark.parametrize(
    ("score", "severity"),
    [
        (0.0, Severity.LOW),
        (0.49, Severity.LOW),
        (0.5, Severity.MEDIUM),
        (0.69, Severity.MEDIUM),
        (0.7, Severity.HIGH),
        (0.89, Severity.HIGH),
        (0.9, Severity.CRITICAL),
        (1.0, Severity.CRITICAL),
    ],
)
def test_severity_score_bands(score, severity):
    assert Severity.from_score(score) is severity


def test_resource_absolute_token_limit_is_strictly_greater_than_limit(store):
    detector = ResourceSpikeDetector(store, absolute_token_limit=100)
    at_limit = event("run-1", "llm", action_type="llm_request", token_count=100)
    above_limit = event("run-1", "llm", action_type="llm_request", token_count=1)

    assert detector.check(at_limit, None) is None
    drift = detector.check(above_limit, None)

    assert drift is not None
    assert drift.detector is DetectorType.RESOURCE_SPIKE
    assert drift.context["current_tokens"] == 101


def test_resource_baseline_threshold_math(store):
    detector = ResourceSpikeDetector(store, spike_multiplier=2.5)
    baseline = BaselineStats(
        agent_id="detector-tests",
        mean_tokens_per_run=100,
        std_tokens_per_run=10,
        is_calibrated=True,
    )
    trace = event("run-1", "llm", action_type="llm_request", token_count=151)
    save(store, trace)

    drift = detector.check(trace, baseline)

    assert drift is not None
    assert drift.context["threshold"] == pytest.approx(125.0)
    assert drift.context["current"] == 151
    assert drift.score == pytest.approx(26 / 125)


def test_resource_baseline_requires_both_threshold_conditions(store):
    detector = ResourceSpikeDetector(store, spike_multiplier=2.5)
    baseline = BaselineStats(
        agent_id="detector-tests",
        mean_tokens_per_run=100,
        std_tokens_per_run=10,
        is_calibrated=True,
    )
    trace = event("run-1", "llm", action_type="llm_request", token_count=125)
    save(store, trace)

    assert detector.check(trace, baseline) is None


def test_resource_counters_survive_more_than_ten_active_runs(store):
    detector = ResourceSpikeDetector(store, absolute_token_limit=100)
    traces = [
        event(f"run-{i}", "llm", action_type="llm_request", token_count=60)
        for i in range(11)
    ]

    for trace in traces:
        assert detector.check(trace, None) is None

    second = event("run-0", "llm", action_type="llm_request", token_count=41)
    drift = detector.check(second, None)

    assert drift is not None
    assert drift.context["current_tokens"] == 101


def test_resource_counters_are_released_at_run_end(store):
    detector = ResourceSpikeDetector(store, absolute_token_limit=100)
    trace = event("run-1", "llm", action_type="llm_request", token_count=60)

    assert detector.check(trace, None) is None
    detector.on_run_end("run-1")

    assert "run-1" not in detector._run_counters
