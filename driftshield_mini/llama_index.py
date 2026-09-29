"""LlamaIndex integration CallbackHandler for LlamaIndex agents."""

from __future__ import annotations

import time
from typing import Any

from driftshield_mini.monitor import DriftMonitor


class DriftLlamaIndexHandler:
    """
    Records tool calls, LLM calls and agent steps for any LlamaIndex agent
    (ReActAgent, OpenAIAgent, Workflow, etc.) via LlamaIndex's callback system.

    Usage:
        from llama_index.core.callbacks import CallbackManager
        from llama_index.core import Settings

        handler = DriftLlamaIndexHandler(
            agent_id="rag-agent-v1",
            alert_webhook="https://hooks.slack.com/...",
            goal_description="Answer questions from the knowledge base",
        )

        Settings.callback_manager = CallbackManager([handler.callback_handler])
        agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)
        agent.chat("What is our refund policy?")
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
        self._tool_starts: dict[str, float] = {}
        self._run_id: str | None = None

    @property
    def callback_handler(self) -> Any:
        """A LlamaIndex BaseCallbackHandler bound to this monitor."""
        try:
            from llama_index.core.callbacks import BaseCallbackHandler
        except ImportError:
            raise ImportError(
                "LlamaIndex integration requires llama-index-core. "
                "Install with: pip install llama-index-core"
            )

        monitor = self.monitor
        outer = self

        class _Handler(BaseCallbackHandler):
            def on_event_start(self, event_type: Any, payload: Any = None,
                               event_id: str = "", **kwargs: Any) -> str:
                if "tool" in str(event_type).lower():
                    outer._tool_starts[event_id or str(time.time())] = time.time()
                elif "llm" in str(event_type).lower() and outer._run_id is None:
                    outer._run_id = monitor.start_run(goal=outer.monitor.goal_drift.goal_description)
                return event_id

            def on_event_end(self, event_type: Any, payload: Any = None,
                             event_id: str = "", **kwargs: Any) -> None:
                et = str(event_type).lower()
                if "tool" in et:
                    started = outer._tool_starts.pop(event_id or "", time.time())
                    tool_name, output = _extract_tool(payload)
                    monitor.record_event(
                        action_type="tool_call",
                        action_name=tool_name,
                        output_data={"result": output[:2000]},
                        duration_ms=(time.time() - started) * 1000,
                        metadata={"framework": "llama_index"},
                    )
                elif "llm" in et:
                    text, tokens = _extract_llm(payload)
                    monitor.record_event(
                        action_type="llm_request",
                        action_name="llamaindex_llm",
                        output_data={"text": text[:2000]},
                        token_count=tokens,
                        metadata={"framework": "llama_index"},
                    )

        return _Handler()

    def start_run(self, goal: str | None = None) -> str:
        self._run_id = self.monitor.start_run(goal=goal)
        return self._run_id

    def end_run(self) -> None:
        if self._run_id:
            self.monitor.end_run(self._run_id)
            self._run_id = None


def _extract_tool(payload: Any) -> tuple[str, str]:
    """Pull tool name + output out of a LlamaIndex tool event payload."""
    tool_name = "llamaindex_tool"
    output = ""
    if isinstance(payload, dict):
        tool_name = str(payload.get("tool_name") or payload.get("name") or tool_name)
        result = payload.get("tool_output") or payload.get("output") or ""
        output = getattr(result, "content", result)
    return tool_name, str(output)


def _extract_llm(payload: Any) -> tuple[str, int]:
    """Pull completion text + token usage out of an LLM payload."""
    text, tokens = "", 0
    if isinstance(payload, dict):
        resp = payload.get("response")
        if resp is not None:
            text = str(getattr(resp, "text", resp))
            usage = getattr(resp, "usage_metadata", None) or getattr(resp, "usage", None)
            if usage is not None:
                total = getattr(usage, "total_tokens", None)
                if total is None and hasattr(usage, "get"):
                    total = usage.get("total_tokens")
                if total:
                    tokens = int(total)
    return text, tokens
