"""AutoGen (Microsoft) integration — monitors ConversableAgent tool calls."""

from __future__ import annotations

import logging
import time
from typing import Any

from driftshield_mini.monitor import DriftMonitor

logger = logging.getLogger(__name__)


class DriftAutogenAgent:
    """
    Wraps an AutoGen ConversableAgent (pyautogen 0.2.x) so every tool
    execution and LLM reply is recorded.

    Usage:
        from autogen import AssistantAgent, UserProxyAgent

        assistant = AssistantAgent("analyst", llm_config=llm_config)
        user = UserProxyAgent("user", human_input_mode="NEVER")

        monitored = DriftAutogenAgent(
            agent=assistant,
            agent_id="autogen-analyst-v1",
            alert_webhook="https://hooks.slack.com/...",
        )
        user.initiate_chat(monitored.agent, message="Summarise Q3 results")
    """

    def __init__(
        self,
        agent: Any,
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
        self._agent = agent
        self._patch_agent(agent)

    @property
    def agent(self) -> Any:
        """The patched AutoGen agent. Pass this to initiate_chat()."""
        return self._agent

    def _patch_agent(self, agent: Any) -> None:
        """Intercept tool execution and LLM replies without changing behaviour."""
        monitor = self.monitor

        # ── Tool calls ──
        original_execute = getattr(agent, "execute_function", None)
        if original_execute:
            def execute_function(function_call: dict, *args: Any, **kwargs: Any):
                name = function_call.get("name", "unknown_tool") if isinstance(function_call, dict) else str(function_call)
                t0 = time.time()
                is_error = False
                result = None
                try:
                    result = original_execute(function_call, *args, **kwargs)
                    return result
                except Exception as e:
                    is_error = True
                    monitor.record_event(
                        action_type="state_transition",
                        action_name=f"tool_error:{name}",
                        output_data={"error": str(e)[:1000]},
                        metadata={"tool_name": name, "framework": "autogen"},
                    )
                    raise
                finally:
                    if not is_error:
                        monitor.record_event(
                            action_type="tool_call",
                            action_name=name,
                            output_data={"result": str(result)[:2000]},
                            duration_ms=(time.time() - t0) * 1000,
                            metadata={"framework": "autogen"},
                        )
            agent.execute_function = execute_function

        # ── LLM replies (token usage) ──
        original_gen = getattr(agent, "generate_oai_reply", None)
        if original_gen:
            def generate_oai_reply(*args: Any, **kwargs: Any):
                t0 = time.time()
                result = original_gen(*args, **kwargs)
                try:
                    content = ""
                    if isinstance(result, tuple) and len(result) == 2:
                        _, response = result
                    else:
                        response = result
                    msg = getattr(response, "message", response)
                    if isinstance(msg, dict):
                        content = str(msg.get("content", ""))[:2000]
                    tokens = len(content) // 4
                    monitor.record_event(
                        action_type="llm_request",
                        action_name="autogen_llm_reply",
                        output_data={"text": content},
                        duration_ms=(time.time() - t0) * 1000,
                        token_count=tokens,
                        metadata={"framework": "autogen"},
                    )
                except Exception:
                    pass
                return result
            agent.generate_oai_reply = generate_oai_reply

    def start_conversation(self, user_proxy: Any, message: str, **kwargs: Any) -> Any:
        """Convenience: start a monitored chat. Goal-drift checks use `message` as the goal."""
        run_id = self.monitor.start_run(goal=message)
        try:
            return user_proxy.initiate_chat(self._agent, message=message, **kwargs)
        finally:
            self.monitor.end_run(run_id)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)
