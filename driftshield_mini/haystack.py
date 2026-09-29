"""Haystack (deepset) integration via a custom tracing span handler."""

from __future__ import annotations

import logging
from typing import Any

from driftshield_mini.monitor import DriftMonitor

logger = logging.getLogger(__name__)


class DriftHaystackTracer:
    """
    Enables Haystack tracing with a span handler that forwards every component
    execution (retrievers, generators, tools) to DriftShield.

    Usage:
        from driftshield_mini.haystack import DriftHaystackTracer

        tracer = DriftHaystackTracer(
            agent_id="haystack-rag-v1",
            alert_webhook="https://hooks.slack.com/...",
            goal_description="Answer policy questions",
        )
        tracer.enable()   # call once, before running pipelines

        # ... run your Haystack pipelines/agents as normal ...
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

    def enable(self) -> None:
        """Turn on Haystack tracing with the DriftShield span handler."""
        try:
            from haystack import tracing
        except ImportError:
            raise ImportError(
                "Haystack integration requires haystack-ai. "
                "Install with: pip install haystack-ai"
            )

        monitor = self.monitor

        class DriftSpanHandler(tracing.DefaultSpanHandler):
            def handle(self, span: Any, component_type: Any) -> None:
                super().handle(span, component_type)  # keep any default hooks
                try:
                    tags = span._tags if hasattr(span, "_tags") else {}
                    name = str(tags.get("haystack.component.name", component_type))
                    raw_input = tags.get("haystack.component.input", {}) or {}
                    input_data = {k: str(v)[:500] for k, v in raw_input.items()}
                    output_data = tags.get("haystack.component.output", {}) or {}
                    is_generator = "generator" in str(component_type).lower() or \
                                   "chat" in str(component_type).lower()

                    tokens = 0
                    usage = tags.get("haystack.component.usage", None) or \
                            (output_data.get("usage", None) if isinstance(output_data, dict) else None)
                    if isinstance(usage, dict):
                        tokens = int(usage.get("total_tokens") or 0)

                    monitor.record_event(
                        action_type="llm_request" if is_generator else "tool_call",
                        action_name=name,
                        input_data=input_data,
                        output_data={"result": str(output_data)[:2000]},
                        token_count=tokens,
                        metadata={"framework": "haystack", "component_type": str(component_type)},
                    )
                except Exception as e:
                    logger.debug(f"DriftShield Haystack span handler error: {e}")

        tracing.enable_tracing(DriftSpanHandler())
        logger.info("DriftShield Haystack tracer enabled")
