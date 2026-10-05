"""Regression tests for DriftShield hardening changes."""

from __future__ import annotations

import threading
import time

import pytest

from driftshield_mini.alerts import AlertDispatcher
from driftshield_mini.models import BaselineStats, DetectorType, DriftEvent, Severity, TraceEvent
from driftshield_mini.monitor import DriftMonitor
from driftshield_mini.storage import TraceStore


@pytest.fixture()
def tmp_db(tmp_path):
    return str(tmp_path / "test.db")


def test_goal_baseline_is_calibrated(monkeypatch, tmp_db):
    class FakeEmbedder:
        def encode(self, text):
            return [1.0, 0.0] if "financial" in str(text).lower() else [0.9, 0.1]

    monkeypatch.setattr(
        "driftshield_mini.baseline.load_embedding_model",
        lambda: FakeEmbedder(),
    )
    m = DriftMonitor(agent_id="goal-test", db_path=tmp_db, calibration_runs=1)
    run = m.start_run(goal="Summarise financial reports")
    m.record_event(
        "llm_request",
        "agent_invoke",
        run_id=run,
        input_data={"input": "Summarise financial reports"},
    )
    m.record_event(
        "llm_request",
        "agent_complete",
        run_id=run,
        output_data={"text": "Financial revenue increased this quarter."},
    )
    m.end_run(run)

    baseline = m.get_baseline()
    assert baseline is not None
    assert baseline.is_calibrated
    assert baseline.mean_goal_similarity > 0


def test_monitor_run_ids_are_thread_local(tmp_db):
    m = DriftMonitor(agent_id="concurrency-test", db_path=tmp_db)
    results = []
    barrier = threading.Barrier(2)

    def worker():
        run_id = m.start_run()
        barrier.wait()
        m.record_event("state_transition", "work")
        results.append(run_id)
        m.end_run(run_id)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    store = TraceStore(db_path=tmp_db)
    run_ids = store.get_run_ids("concurrency-test", limit=10)
    assert len(set(results)) == 2
    assert set(results).issubset(set(run_ids))


def test_failed_webhook_can_alert_again(monkeypatch):
    dispatcher = AlertDispatcher(
        webhook_url="https://example.invalid/hook",
        cooldown_seconds=60,
    )
    event = DriftEvent(
        agent_id="a",
        run_id="r",
        detector=DetectorType.ACTION_LOOP,
        severity=Severity.HIGH,
        score=0.8,
        message="loop",
        suggested_action="inspect",
    )

    class Response:
        def raise_for_status(self):
            raise RuntimeError("network failure")

    class Client:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr("httpx.Client", lambda **kwargs: Client())

    assert dispatcher.send_sync(event) is False
    assert dispatcher.should_alert(event) is True
    dispatcher.close()


def test_background_alert_submission_does_not_block(monkeypatch):
    dispatcher = AlertDispatcher(webhook_url="https://example.invalid/hook")
    event = DriftEvent(
        agent_id="a",
        run_id="r",
        detector=DetectorType.ACTION_LOOP,
        severity=Severity.HIGH,
        score=0.8,
        message="loop",
        suggested_action="inspect",
    )

    started = threading.Event()

    def slow_send(_event):
        started.set()
        time.sleep(0.2)
        return True

    monkeypatch.setattr(dispatcher, "_send_sync", slow_send)
    future = dispatcher.send_background(event)
    assert started.wait(0.5)
    assert future is not None
    dispatcher.close()


def test_resource_counter_cleanup_uses_start_time(tmp_db):
    m = DriftMonitor(agent_id="resource-test", db_path=tmp_db)
    detector = m.resource_spike
    detector._run_counters = {
        "z-newer": {
            "total_tokens": 0,
            "total_duration_ms": 0.0,
            "tool_calls": 0,
            "llm_calls": 0,
            "start_time": 20.0,
        },
        "a-older": {
            "total_tokens": 0,
            "total_duration_ms": 0.0,
            "tool_calls": 0,
            "llm_calls": 0,
            "start_time": 10.0,
        },
    }
    for index in range(8):
        detector._run_counters[f"run-{index}"] = {
            "total_tokens": 0,
            "total_duration_ms": 0.0,
            "tool_calls": 0,
            "llm_calls": 0,
            "start_time": 30.0 + index,
        }

    detector._get_run_counter("new-run")

    assert len(detector._run_counters) == 10
    assert "a-older" not in detector._run_counters
    assert "z-newer" in detector._run_counters


def test_goal_detection_isolated_between_concurrent_runs(monkeypatch, tmp_db):
    class FakeEmbedder:
        def encode(self, text):
            value = str(text).lower()
            if "alpha" in value:
                return [1.0, 0.0]
            if "beta" in value:
                return [0.0, 1.0]
            return [1.0, 1.0]

    monkeypatch.setattr(
        "driftshield_mini.detectors.goal_drift.load_embedding_model",
        lambda: FakeEmbedder(),
    )
    m = DriftMonitor(
        agent_id="goal-concurrency",
        db_path=tmp_db,
        similarity_threshold=0.95,
    )
    barrier = threading.Barrier(2)
    results = []

    def worker(goal, output):
        run_id = m.start_run(goal=goal)
        barrier.wait()
        results.append(m.record_event(
            "llm_request",
            "complete",
            run_id=run_id,
            output_data={"text": output},
        ))

    threads = [
        threading.Thread(target=worker, args=("Alpha task", "Alpha task completed successfully.")),
        threading.Thread(target=worker, args=("Beta task", "Beta task completed successfully.")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == [[], []]
    m.close()


def test_resource_counter_updates_are_thread_safe(tmp_db):
    m = DriftMonitor(agent_id="resource-concurrency", db_path=tmp_db)
    detector = m.resource_spike
    barrier = threading.Barrier(20)

    def worker():
        barrier.wait()
        detector.check(
            TraceEvent(
                agent_id="resource-concurrency",
                run_id="shared-run",
                action_type="llm_request",
                action_name="step",
                token_count=1,
            ),
            BaselineStats(agent_id="resource-concurrency"),
        )

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert detector._run_counters["shared-run"]["total_tokens"] == 20
    assert detector._run_counters["shared-run"]["llm_calls"] == 20
    m.close()


def test_detector_exception_is_logged_and_does_not_break_monitor(caplog, tmp_db):
    class ExplodingDetector:
        def name(self):
            return "test_detector"

        def check(self, event, baseline):
            raise RuntimeError("detector exploded")

    m = DriftMonitor(agent_id="exception-observability", db_path=tmp_db)
    m._detectors = [ExplodingDetector()]

    with caplog.at_level("ERROR"):
        assert m.record_event("state_transition", "work") == []

    assert "Detector 'test_detector' failed: detector exploded" in caplog.text
    assert len(m.store.get_traces(agent_id="exception-observability")) == 1
    m.close()
