"""Google ADK (Agent Development Kit) integration via agent callbacks."""

from __future__ import annotations

import logging
import time
from typing import Any

from driftshield_mini.monitor import DriftMonitor

logger = logging.getLogger(__name__)


class DriftADKCallbacks:
    """
    Provides `before_tool_callback` / `after_tool_callback` functions for a
    Google ADK Agent, giving tool-level + run-level monitoring.

    Usage:
        from google.adk.agents import Agent
        from driftshield_mini.google_adk import DriftADKCallbacks

        drift = DriftADKCallbacks(
            agent_id="adk-support-agent-v1",
            alert_webhook="https://hooks.slack.com/...",
            goal_description="Resolve customer support tickets",
        )

        agent = Agent(
            model="gemini-2.5-flash",
            name="support_agent",
            instruction="...",
            tools=[lookup_order, refund_order],
            before_tool_callback=drift.before_tool,
            after_tool_callback=drift.after_tool,
        )
    """

    def __init__(
        self,
        agent_id: str,
        alert_webhook: str | None = None,
        goal_description: str = "",
        calibration_runs: int = 30,
        db_path: str | None = None,
        **monitor_kwargs: Any,
    ):
        self.monitor = DriftMonitor(
            agent_id=agent_id,
            alert_webhook=alert_webhook,
            goal_description=goal_description,
            calibration_runs=calibration_runs,
            db_path=db_path,
            **monitor_kwargs,
        )
        self._starts: dict[int, float] = {}

    #  ADK callbacks

    def before_tool(self, tool: Any, tool_args: dict, tool_context: Any) -> None:
        """ADK before_tool_callback. Return None to let the tool run normally."""
        self._starts[id(tool_context)] = time.time()
        name = getattr(tool, "__name__", None) or getattr(tool, "name", None) or str(tool)
        self.monitor.record_event(
            action_type="state_transition",
            action_name=f"tool_started:{name}",
            input_data={k: str(v)[:500] for k, v in (tool_args or {}).items()},
            metadata={"framework": "google_adk", "tool_name": name},
        )


    def after_tool(self, tool: Any, tool_args: dict, tool_context: Any, tool_response: Any) -> Any:
        """ADK after_tool_callback. Return the response unchanged."""
        name = getattr(tool, "__name__", None) or getattr(tool, "name", None) or str(tool)
        started = self._starts.pop(id(tool_context), time.time())
        self.monitor.record_event(
            action_type="tool_call",
            action_name=name,
            output_data={"result": str(tool_response)[:2000]},
            duration_ms=(time.time() - started) * 1000,
            metadata={"framework": "google_adk"},
        )
        return tool_response  # pass the result through untouched

    # ── Optional run lifecycle helper ─────────────────────────────

    def wrap_run(self, run_coro: Any, goal: str | None = None) -> Any:
        """Wrap an ADK `runner.run(...)` async generator with a monitored run.

        Usage:
            response = drift.wrap_run(
                runner.run(user_id="u1", session_id="s1", new_message=msg),
                goal="Resolve ticket #1234",
            )
            async for event in response:
                ...
        """
        async def _wrapped():
            run_id = self.monitor.start_run(goal=goal or "")
            try:
                async for event in run_coro:
                    yield event
            finally:
                self.monitor.end_run(run_id)

        return _wrapped()
