"""OpenAI Assistants / Responses API integration."""

from __future__ import annotations

import time
from typing import Any

from driftshield_mini.monitor import DriftMonitor


class DriftOpenAIClient:
    """
    Wraps an `openai.OpenAI` client so Assistants runs are monitored end-to-end:
    run start, every tool call step, token usage, and the final output.

    Usage:
        import openai
        from driftshield_mini.openai_assistants import DriftOpenAIClient

        client = DriftOpenAIClient(
            openai.OpenAI(),
            agent_id="assistant-invoice-v1",
            alert_webhook="https://hooks.slack.com/...",
            goal_description="Process customer invoices",
        )

        run = client.run_assistant(thread_id=thread.id, assistant_id=assistant.id)
        messages = client.beta.threads.messages.list(thread_id=thread.id)
    """

    def __init__(
        self,
        client: Any,
        agent_id: str,
        alert_webhook: str | None = None,
        goal_description: str = "",
        calibration_runs: int = 30,
        db_path: str | None = None,
        **monitor_kwargs: Any,
    ):
        self._client = client
        self.monitor = DriftMonitor(
            agent_id=agent_id,
            alert_webhook=alert_webhook,
            goal_description=goal_description,
            calibration_runs=calibration_runs,
            db_path=db_path,
            **monitor_kwargs,
        )

    def run_assistant(
        self,
        thread_id: str,
        assistant_id: str,
        instructions: str | None = None,
        additional_instructions: str | None = None,
        poll_interval: float = 1.0,
        **kwargs: Any,
    ) -> Any:
        """Create a run, poll it to completion, monitoring every step."""
        run_id = self.monitor.start_run(goal=instructions or additional_instructions or "")
        t0 = time.time()
        try:
            run = self._client.beta.threads.runs.create(
                thread_id=thread_id, assistant_id=assistant_id,
                instructions=instructions, additional_instructions=additional_instructions,
                **kwargs,
            )
            run = self._poll_and_monitor(run, thread_id, poll_interval)

            usage = getattr(run, "usage", None)
            tokens = int(getattr(usage, "total_tokens", 0) or 0)
            output_text = self._collect_output(thread_id)
            self.monitor.record_event(
                action_type="llm_request",
                action_name="assistant_run_complete",
                run_id=run_id,
                output_data={"text": output_text[:2000]},
                duration_ms=(time.time() - t0) * 1000,
                token_count=tokens,
                metadata={"framework": "openai_assistants", "status": getattr(run, "status", "")},
            )
            return run
        except Exception as e:
            self.monitor.record_event(
                action_type="state_transition",
                action_name="assistant_run_error",
                run_id=run_id,
                output_data={"error": str(e)[:1000]},
                duration_ms=(time.time() - t0) * 1000,
            )
            raise
        finally:
            self.monitor.end_run(run_id)

    # ── Internals ─────────────────────────────────────────────────

    def _poll_and_monitor(self, run: Any, thread_id: str, poll_interval: float) -> Any:
        """Poll the run; record each step's tool calls as they appear."""
        terminal = {"completed", "failed", "cancelled", "expired", "incomplete"}
        while getattr(run, "status", "") not in terminal:
            time.sleep(poll_interval)
            run = self._client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
            self._record_steps(thread_id, run.id)

        if run.status != "completed":
            raise RuntimeError(
                f"Assistant run ended with status '{run.status}': {getattr(run, 'last_error', '')}"
            )
        self._record_steps(thread_id, run.id)
        return run

    def _record_steps(self, thread_id: str, openai_run_id: str) -> None:
        """Record any not-yet-seen tool-call steps for this run."""
        try:
            steps = self._client.beta.threads.runs.steps.list(
                thread_id=thread_id, run_id=openai_run_id, limit=100
            )
            for step in reversed(list(steps.data)):
                key = f"openai_step:{step.id}"
                if key in self._seen_steps:
                    continue
                self._seen_steps.add(key)
                if step.type == "tool_calls":
                    for tc in step.step_details.tool_calls:
                        self.monitor.record_event(
                            action_type="tool_call",
                            action_name=getattr(tc, "type", "tool_call"),
                            output_data={"tool_call_id": getattr(tc, "id", "")},
                            metadata={"framework": "openai_assistants", "step_id": step.id},
                        )
        except Exception:
            pass  # Step listing is best-effort; never break the run

    @property
    def _seen_steps(self) -> set[str]:
        if not hasattr(self, "_seen_steps_set"):
            self._seen_steps_set = set()
        return self._seen_steps_set

    def _collect_output(self, thread_id: str) -> str:
        try:
            msgs = self._client.beta.threads.messages.list(thread_id=thread_id, limit=5)
            parts = []
            for m in msgs.data:
                if m.role == "assistant":
                    for c in m.content:
                        text = getattr(getattr(c, "text", None), "value", None)
                        if text:
                            parts.append(text)
            return "\n".join(parts)
        except Exception:
            return ""

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
