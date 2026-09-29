"""Tests for the framework wrappers run: python -m pytest tests/test_wrappers.py -v"""

from __future__ import annotations

import json

import pytest

from driftshield_mini.export import export_drift_events, export_traces
from driftshield_mini.monitor import DriftMonitor
from driftshield_mini.storage import TraceStore


@pytest.fixture()
def tmp_db(tmp_path):
    return str(tmp_path / "test.db")


def _make_monitor(tmp_db, **kw):
    return DriftMonitor(
        agent_id="test-agent",
        db_path=tmp_db,
        goal_description="Summarise financial reports",
        **kw,
    )


def test_record_event_stores_trace(tmp_db):
    m = _make_monitor(tmp_db)
    m.record_event(action_type="tool_call", action_name="search_db", run_id="r1")
    store = TraceStore(db_path=tmp_db)
    traces = store.get_run_traces("test-agent", "r1")
    assert len(traces) == 1
    assert traces[0].action_name == "search_db"


def test_loop_detection_fires(tmp_db):
    m = _make_monitor(tmp_db, loop_max_repeats=3, calibration_runs=1)
    events = []
    for i in range(5):
        events.extend(m.record_event(action_type="tool_call", action_name="same_tool", run_id="r1"))
    assert any(e.detector.value == "action_loop" for e in events)


def test_export_csv(tmp_db, tmp_path):
    m = _make_monitor(tmp_db)
    m.record_event(action_type="tool_call", action_name="t", run_id="r1")
    out = tmp_path / "audit.csv"
    export_traces(TraceStore(db_path=tmp_db), out, agent_id="test-agent")
    content = out.read_text()
    assert "action_name" in content.splitlines()[0]
    assert "tool_call" in content


def test_export_json(tmp_db, tmp_path):
    m = _make_monitor(tmp_db)
    for _ in range(5):
        m.record_event(action_type="tool_call", action_name="x", run_id="r1")
    out = tmp_path / "audit.json"
    export_traces(TraceStore(db_path=tmp_db), out, agent_id="test-agent", fmt="json")
    data = json.loads(out.read_text())
    assert len(data) == 5


def test_driftcrew_importable_without_crewai(tmp_db):
    from driftshield_mini.crewai import DriftCrew

    class FakeCrew:
        description = "test goal"

        def kickoff(self, **kw):
            return "done"

    crew = DriftCrew(crew=FakeCrew(), agent_id="test-agent", db_path=tmp_db)
    assert crew.kickoff() == "done"
    store = TraceStore(db_path=tmp_db)
    assert store.get_run_ids("test-agent"), "kickoff should have created a run"


def test_autogen_wrapper_records_tool_calls(tmp_db):
    from driftshield_mini.autogen import DriftAutogenAgent

    class FakeAutogenAgent:
        def execute_function(self, function_call, *a, **k):
            return {"content": "ok"}

        def generate_oai_reply(self, *a, **k):
            return True, {"content": "hello world"}

    wrapped = DriftAutogenAgent(FakeAutogenAgent(), agent_id="autogen-test", db_path=tmp_db)
    wrapped.agent.execute_function({"name": "get_price"})
    store = TraceStore(db_path=tmp_db)
    traces = store.get_traces(agent_id="autogen-test")
    assert any(t.action_type == "tool_call" and t.action_name == "get_price" for t in traces)


def test_adk_callbacks_record_events(tmp_db):
    from driftshield_mini.google_adk import DriftADKCallbacks

    drift = DriftADKCallbacks(agent_id="adk-test", db_path=tmp_db)

    def fake_tool(order_id: str):
        return {"status": "refunded"}

    class Ctx:
        pass

    ctx = Ctx()
    drift.before_tool(fake_tool, {"order_id": "42"}, ctx)
    result = drift.after_tool(fake_tool, {"order_id": "42"}, ctx, {"status": "refunded"})
    assert result == {"status": "refunded"}
    store = TraceStore(db_path=tmp_db)
    traces = store.get_traces(agent_id="adk-test")
    assert any(t.action_type == "tool_call" for t in traces)
