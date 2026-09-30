"""Reproducible detector benchmark and security regression tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from driftshield_mini import DriftMonitor
from driftshield_mini.models import DetectorType


@dataclass
class Case:
    name: str
    expected: bool
    action_names: list[str]
    goal: str = ""
    output: str = ""


def run_case(tmp_path: Path, case: Case) -> set[DetectorType]:
    monitor = DriftMonitor(
        agent_id=case.name,
        db_path=str(tmp_path / f"{case.name}.db"),
        goal_description=case.goal,
        calibration_runs=999,
        loop_max_repeats=4,
    )
    run_id = monitor.start_run()
    for i, action in enumerate(case.action_names):
        monitor.record_event(
            "tool_call",
            action,
            run_id=run_id,
            input_data={"index": i},
        )
    if case.output:
        monitor.record_event(
            "llm_request",
            "agent_complete",
            run_id=run_id,
            input_data={"input": case.goal},
            output_data={"text": case.output},
        )
    detected = {event.detector for event in monitor.get_recent_alerts(limit=100)}
    monitor.end_run(run_id)
    monitor.close()
    return detected


@pytest.mark.parametrize(
    "case",
    [
        Case("true_single_loop", True, ["search"] * 6),
        Case("true_cycle", True, ["search", "format"] * 4),
        Case("legitimate_repetition", False, ["search"] * 3),
        Case("legitimate_sequence", False, ["search", "format"] * 2),
    ],
)
def test_action_loop_cases(tmp_path, case):
    detected = run_case(tmp_path, case)
    assert (DetectorType.ACTION_LOOP in detected) is case.expected


def test_goal_preserving_output_does_not_use_unrelated_goal(tmp_path):
    case = Case(
        "goal_preserving",
        False,
        [],
        goal="Summarise financial reports",
        output="The financial report shows revenue increased during the quarter.",
    )
    detected = run_case(tmp_path, case)
    assert DetectorType.GOAL_DRIFT not in detected


def test_goal_drift_case(tmp_path, monkeypatch):
    class FakeEmbedder:
        def encode(self, text):
            text = str(text).lower()
            return [1.0, 0.0] if "financial" in text else [0.0, 1.0]

    monkeypatch.setattr(
        "driftshield_mini.detectors.goal_drift.load_embedding_model",
        lambda: FakeEmbedder(),
    )
    case = Case(
        "goal_drift",
        True,
        [],
        goal="Summarise financial reports",
        output="Explain how to grow tomatoes in a garden.",
    )
    detected = run_case(tmp_path, case)
    assert DetectorType.GOAL_DRIFT in detected


def test_absolute_resource_limit(tmp_path):
    monitor = DriftMonitor(
        agent_id="resource-limit",
        db_path=str(tmp_path / "resource.db"),
    )
    run_id = monitor.start_run()
    events = monitor.record_event(
        "llm_request",
        "large_output",
        run_id=run_id,
        token_count=60_000,
    )
    monitor.close()
    assert any(e.detector == DetectorType.RESOURCE_SPIKE for e in events)


def test_trace_json_is_bounded_by_application_contract(tmp_path):
    # The core store accepts arbitrary JSON, so callers should apply their own
    # retention/redaction policy. This test documents that no plaintext secret
    # should be added by DriftShield itself.
    monitor = DriftMonitor(agent_id="security", db_path=str(tmp_path / "s.db"))
    run_id = monitor.start_run()
    monitor.record_event("state_transition", "test", run_id=run_id, metadata={"safe": True})
    assert monitor.store.get_run_traces("security", run_id)
    monitor.close()
