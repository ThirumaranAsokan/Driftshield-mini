"""CrewAI integration — DriftCrew wrapper for CrewAI crews."""

from __future__ import annotations

import time
from typing import Any

from driftshield.monitor import DriftMonitor


class DriftCrew:
    """
    Wraps a CrewAI crew with DriftShield monitoring.

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

    def kickoff(self, **kwargs: Any) -> Any:
        """Wrap CrewAI's kickoff method with drift monitoring."""
        # Use crew description as goal if available
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
