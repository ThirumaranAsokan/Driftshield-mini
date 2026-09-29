"""CrewAI integration — DriftCrew wrapper with full tool-level visibility."""

from __future__ import annotations

import logging
import time
from typing import Any

from driftshield_mini.monitor import DriftMonitor

logger = logging.getLogger(__name__)


class DriftCrew:
    """
    Wraps a CrewAI crew with DriftShield monitoring.

    Records:
      * run start / completion (kickoff lifecycle)
      * every tool call inside the crew (via the CrewAI event bus)
      * every LLM call with token usage where the event exposes it
      * errors

    Usage:
        crew = DriftCrew(
            crew=existing_crew,
            agent_id="research-team-v1",
            alert_webhook="https://discord.com/api/webhooks/...",
        )
        result = crew.kickoff()
    """

    def __init__(
        self,
        crew: Any,
        agent_id: str,
        alert_webhook: str | None = None,
        goal_description: str = "",
        calibration_runs: int = 30,
        db_path: str | None = None,
        **monitor_kwargs: Any,
    ):
        self._crew = crew
        self.monitor = DriftMonitor(
            agent_id=agent_id,
            alert_webhook=alert_webhook,
            goal_description=goal_description,
            calibration_runs=calibration_runs,
            db_path=db_path,
            **monitor_kwargs,
        )
        self._tool_start_times: dict[str, float] = {}
        self._subscribe_to_events()

    # ── Event bus subscription ────────────────────────────────────

    def _subscribe_to_events(self) -> None:
        """Hook into CrewAI's event bus so tool/LLM calls are visible.

        Uses defensive imports — if the installed CrewAI version doesn't
        expose these events, the wrapper still works at kickoff level.
        """
        try:
            from crewai.utilities.events import crewai_event_bus
            from crewai.utilities.events.tool_usage_events import (
                ToolUsageErrorEvent,
                ToolUsageFinishedEvent,
                ToolUsageStartedEvent,
            )
            from crewai.utilities.events.llm_events import (
                LLMCallCompletedEvent,
                LLMCallStartedEvent,
            )
        except ImportError:
            logger.warning(
                "CrewAI event bus not available in this version. "
                "Falling back to kickoff-level monitoring only. "
                "Upgrade crewai for tool-level traces."
            )
            return

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def _on_tool_started(source: Any, event: Any) -> None:
            name = getattr(event, "tool_name", None) or getattr(source, "name", "unknown_tool")
            key = f"{name}:{id(event)}"
            self._tool_start_times[key] = time.time()
            self.monitor.record_event(
                action_type="state_transition",
                action_name=f"tool_started:{name}",
                metadata={"tool_name": name},
            )

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def _on_tool_finished(source: Any, event: Any) -> None:
            name = getattr(event, "tool_name", None) or getattr(source, "name", "unknown_tool")
            key = f"{name}:{id(event)}"
            started = self._tool_start_times.pop(key, time.time())
            output = str(getattr(event, "result", "") or "")[:2000]
            self.monitor.record_event(
                action_type="tool_call",
                action_name=name,
                output_data={"result": output},
                duration_ms=(time.time() - started) * 1000,
                metadata={"framework": "crewai"},
            )

        @crewai_event_bus.on(ToolUsageErrorEvent)
        def _on_tool_error(source: Any, event: Any) -> None:
            name = getattr(event, "tool_name", None) or getattr(source, "name", "unknown_tool")
            self.monitor.record_event(
                action_type="state_transition",
                action_name=f"tool_error:{name}",
                output_data={"error": str(getattr(event, "error", "unknown"))[:1000]},
                metadata={"tool_name": name},
            )

        @crewai_event_bus.on(LLMCallStartedEvent)
        def _on_llm_started(source: Any, event: Any) -> None:
            self._tool_start_times[f"llm:{id(event)}"] = time.time()

        @crewai_event_bus.on(LLMCallCompletedEvent)
        def _on_llm_completed(source: Any, event: Any) -> None:
            started = self._tool_start_times.pop(f"llm:{id(event)}", time.time())
            usage = getattr(event, "token_usage", None) or getattr(source, "token_usage", None)
            tokens = 0
            if isinstance(usage, dict):
                tokens = int(usage.get("total_tokens") or 0)
            response = str(getattr(event, "response", "") or "")[:2000]
            self.monitor.record_event(
                action_type="llm_request",
                action_name="crew_llm_call",
                output_data={"text": response},
                duration_ms=(time.time() - started) * 1000,
                token_count=tokens,
                metadata={"framework": "crewai"},
            )

    # ── kickoff lifecycle ─────────────────────────────────────────

    def kickoff(self, **kwargs: Any) -> Any:
        """Wrap CrewAI's kickoff method with drift monitoring."""
        goal = ""
        if hasattr(self._crew, "description"):
            goal = self._crew.description or ""
        elif hasattr(self._crew, "tasks") and self._crew.tasks:
            first_task = self._crew.tasks[0]
            goal = getattr(first_task, "description", "")

        run_id = self.monitor.start_run(goal=goal)
        start = time.time()

        try:
            self.monitor.record_event(
                action_type="llm_request",
                action_name="crew_kickoff",
                run_id=run_id,
                input_data={"kwargs": str(kwargs)},
            )

            result = self._crew.kickoff(**kwargs)

            elapsed_ms = (time.time() - start) * 1000
            output_text = str(result) if result else ""

            self.monitor.record_event(
                action_type="llm_request",
                action_name="crew_complete",
                run_id=run_id,
                output_data={"text": output_text},
                duration_ms=elapsed_ms,
                token_count=len(output_text) // 4,
            )

            return result

        except Exception as e:
            self.monitor.record_event(
                action_type="state_transition",
                action_name="crew_error",
                run_id=run_id,
                output_data={"error": str(e)},
                duration_ms=(time.time() - start) * 1000,
            )
            raise
        finally:
            self.monitor.end_run(run_id)

    def __getattr__(self, name: str) -> Any:
        """Proxy to underlying crew."""
        return getattr(self._crew, name)
