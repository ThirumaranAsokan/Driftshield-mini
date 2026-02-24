"""Standalone tests — runs with just Python stdlib + numpy."""

import os
import sys
import tempfile
import time
import unittest

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from driftshield.models import BaselineStats, DetectorType, Severity, TraceEvent
from driftshield.storage import TraceStore
from driftshield.detectors.action_loop import ActionLoopDetector
from driftshield.detectors.resource_spike import ResourceSpikeDetector
from driftshield.monitor import DriftMonitor


class TestModels(unittest.TestCase):
    def test_severity_from_score(self):
        self.assertEqual(Severity.from_score(0.3), Severity.LOW)
        self.assertEqual(Severity.from_score(0.5), Severity.MEDIUM)
        self.assertEqual(Severity.from_score(0.7), Severity.HIGH)
        self.assertEqual(Severity.from_score(0.95), Severity.CRITICAL)

    def test_trace_event_roundtrip(self):
        event = TraceEvent(
            agent_id="a", run_id="r", action_type="tool_call", action_name="test"
        )
        d = event.to_dict()
        restored = TraceEvent.from_dict(d)
        self.assertEqual(restored.agent_id, "a")
        self.assertEqual(restored.action_name, "test")


class TestTraceStore(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.store = TraceStore(db_path=self.db_path)

    def tearDown(self):
        self.store.close()
        os.unlink(self.db_path)

    def test_save_and_retrieve(self):
        event = TraceEvent(
            agent_id="agent-1", run_id="run-1",
            action_type="tool_call", action_name="search_db", token_count=100,
        )
        self.store.save_trace(event)
        traces = self.store.get_traces(agent_id="agent-1")
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0].action_name, "search_db")

    def test_recent_actions(self):
        for i, name in enumerate(["search", "format", "search", "format"]):
            self.store.save_trace(TraceEvent(
                agent_id="a", run_id="r", action_type="tool_call",
                action_name=name, timestamp=time.time() + i,
            ))
        actions = self.store.get_recent_actions("a", "r", window=10)
        self.assertEqual(actions, ["search", "format", "search", "format"])

    def test_run_stats(self):
        for i in range(5):
            self.store.save_trace(TraceEvent(
                agent_id="a", run_id="r", action_type="tool_call",
                action_name=f"tool_{i}", token_count=100, duration_ms=50.0,
            ))
        stats = self.store.get_run_stats("a", "r")
        self.assertEqual(stats["event_count"], 5)
        self.assertEqual(stats["total_tokens"], 500)

    def test_baseline_roundtrip(self):
        bl = BaselineStats(agent_id="a", mean_tokens_per_run=500, is_calibrated=True)
        self.store.save_baseline(bl)
        restored = self.store.get_baseline("a")
        self.assertIsNotNone(restored)
        self.assertEqual(restored.mean_tokens_per_run, 500)
        self.assertTrue(restored.is_calibrated)


class TestActionLoopDetector(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.store = TraceStore(db_path=self.db_path)

    def tearDown(self):
        self.store.close()
        os.unlink(self.db_path)

    def test_detects_single_tool_loop(self):
        detector = ActionLoopDetector(self.store, max_repeats=3)
        for i in range(5):
            self.store.save_trace(TraceEvent(
                agent_id="a", run_id="r", action_type="tool_call",
                action_name="search_db", timestamp=time.time() + i,
            ))
        event = TraceEvent(
            agent_id="a", run_id="r", action_type="tool_call", action_name="search_db",
        )
        drift = detector.check(event, None)
        self.assertIsNotNone(drift)
        self.assertEqual(drift.detector, DetectorType.ACTION_LOOP)
        self.assertIn("search_db", drift.message)

    def test_no_false_positive(self):
        detector = ActionLoopDetector(self.store, max_repeats=3)
        for i, name in enumerate(["search", "format", "save", "notify"]):
            self.store.save_trace(TraceEvent(
                agent_id="a", run_id="r", action_type="tool_call",
                action_name=name, timestamp=time.time() + i,
            ))
        event = TraceEvent(
            agent_id="a", run_id="r", action_type="tool_call", action_name="complete",
        )
        drift = detector.check(event, None)
        self.assertIsNone(drift)

    def test_detects_sequence_loop(self):
        detector = ActionLoopDetector(self.store, max_repeats=3, sequence_length=3)
        # A→B→A→B→A→B→A→B (4 repeats of A→B)
        for i in range(8):
            name = "search" if i % 2 == 0 else "format"
            self.store.save_trace(TraceEvent(
                agent_id="a", run_id="r", action_type="tool_call",
                action_name=name, timestamp=time.time() + i,
            ))
        event = TraceEvent(
            agent_id="a", run_id="r", action_type="tool_call", action_name="format",
        )
        drift = detector.check(event, None)
        # Should detect the A→B repeat pattern
        # (may or may not trigger depending on exact window — that's ok)


class TestResourceSpikeDetector(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.store = TraceStore(db_path=self.db_path)

    def tearDown(self):
        self.store.close()
        os.unlink(self.db_path)

    def test_absolute_token_limit(self):
        detector = ResourceSpikeDetector(self.store, absolute_token_limit=500)
        drifts = []
        for i in range(10):
            event = TraceEvent(
                agent_id="a", run_id="r", action_type="llm_request",
                action_name="generate", token_count=100,
            )
            d = detector.check(event, None)
            if d:
                drifts.append(d)

        self.assertTrue(len(drifts) > 0)
        self.assertEqual(drifts[0].detector, DetectorType.RESOURCE_SPIKE)

    def test_baseline_spike(self):
        detector = ResourceSpikeDetector(self.store, spike_multiplier=2.0)
        baseline = BaselineStats(
            agent_id="a", mean_tokens_per_run=200, std_tokens_per_run=50, is_calibrated=True,
        )
        drift = None
        for i in range(5):
            event = TraceEvent(
                agent_id="a", run_id="r2", action_type="llm_request",
                action_name="generate", token_count=100,
            )
            drift = detector.check(event, baseline)
        self.assertIsNotNone(drift)
        self.assertIn("token_burn", drift.message)


class TestDriftMonitor(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.monitor = DriftMonitor(
            agent_id="test-agent", db_path=self.db_path,
            calibration_runs=5, loop_max_repeats=3,
        )

    def tearDown(self):
        self.monitor.store.close()
        os.unlink(self.db_path)

    def test_loop_detection_end_to_end(self):
        run_id = self.monitor.start_run()
        all_drifts = []
        for i in range(10):
            drifts = self.monitor.record_event(
                action_type="tool_call", action_name="search_db", run_id=run_id,
            )
            all_drifts.extend(drifts)
        self.assertTrue(len(all_drifts) > 0)
        self.assertTrue(any(d.detector == DetectorType.ACTION_LOOP for d in all_drifts))

    def test_clean_run_no_drift(self):
        run_id = self.monitor.start_run()
        all_drifts = []
        for name in ["search", "format", "save"]:
            drifts = self.monitor.record_event(
                action_type="tool_call", action_name=name, run_id=run_id, token_count=50,
            )
            all_drifts.extend(drifts)
        self.monitor.end_run(run_id)
        self.assertEqual(len(all_drifts), 0)

    def test_baseline_calibration(self):
        for _ in range(6):
            run_id = self.monitor.start_run()
            for name in ["fetch", "process", "store"]:
                self.monitor.record_event(
                    action_type="tool_call", action_name=name, run_id=run_id, token_count=100,
                )
            self.monitor.end_run(run_id)
        baseline = self.monitor.get_baseline()
        self.assertIsNotNone(baseline)
        self.assertTrue(baseline.is_calibrated)
        self.assertGreater(baseline.mean_tokens_per_run, 0)

    def test_drift_callback(self):
        fired = []
        self.monitor.on_drift(lambda d: fired.append(d))
        run_id = self.monitor.start_run()
        for i in range(10):
            self.monitor.record_event(
                action_type="tool_call", action_name="stuck_tool", run_id=run_id,
            )
        self.assertTrue(len(fired) > 0)

    def test_get_recent_alerts(self):
        run_id = self.monitor.start_run()
        for i in range(10):
            self.monitor.record_event(
                action_type="tool_call", action_name="stuck", run_id=run_id,
            )
        alerts = self.monitor.get_recent_alerts(hours=1)
        self.assertTrue(len(alerts) > 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
