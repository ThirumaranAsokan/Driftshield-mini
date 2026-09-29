"""Semantic Kernel (Microsoft) integration via function invocation filters."""

from __future__ import annotations

import logging
import time
from typing import Any

from driftshield_mini.monitor import DriftMonitor

logger = logging.getLogger(__name__)


class DriftKernelFilter:
    """
    A Semantic Kernel function-invocation filter that records every plugin
    function call (including AI service calls) with arguments and results.

    Usage:
        from semantic_kernel import Kernel
        from driftshield_mini.semantic_kernel import DriftKernelFilter

        kernel = Kernel()
        # ... add plugins / AI services ...

        drift_filter = DriftKernelFilter(
            agent_id="sk-invoice-agent-v1",
            alert_webhook="https://hooks.slack.com/...",
            goal_description="Reconcile invoices",
        )
        drift_filter.apply_to_kernel(kernel)

        result = await kernel.invoke(plugin_name="Invoices", function_name="reconcile")
    """

    def __init__(
        self,
        agent_id: str,
        alert_webhook: str | None = None,
        goal_description: str = "",
        calibration_runs: int = 30,
        db_path: str | None = None,
        track_ai_service_calls: bool = True,
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
        self.track_ai_service_calls = track_ai_service_calls

    def apply_to_kernel(self, kernel: Any) -> Any:
        """Register the filter on a Kernel instance. Returns the kernel."""
        try:
            from semantic_kernel.filters import FilterTypes
        except ImportError:
            raise ImportError(
                "Semantic Kernel integration requires semantic-kernel. "
                "Install with: pip install semantic-kernel"
            )

        kernel.add_filter(FilterTypes.FUNCTION_INVOCATION, self._function_filter)
        logger.info("DriftShield Semantic Kernel filter applied")
        return kernel

    async def _function_filter(self, context: Any, next_func: Any) -> Any:
        """SK filter: record before/after each function invocation."""
        if hasattr(context.function, "plugin_name"):
            function_name = f"{context.function.plugin_name}.{context.function.name}"
        else:
            function_name = str(getattr(context.function, "name", "unknown"))

        metadata = getattr(context.function, "metadata", None)
        is_ai = bool(getattr(metadata, "is_prompt", False))

        if is_ai and not self.track_ai_service_calls:
            return await next_func(context)

        args = {}
        try:
            for k, v in context.arguments.items():
                args[k] = str(v)[:500]
        except Exception:
            pass

        self.monitor.record_event(
            action_type="state_transition",
            action_name=f"function_started:{function_name}",
            input_data=args,
            metadata={"framework": "semantic_kernel", "is_ai": is_ai},
        )

        t0 = time.time()
        try:
            await next_func(context)
        except Exception as e:
            self.monitor.record_event(
                action_type="state_transition",
                action_name=f"function_error:{function_name}",
                output_data={"error": str(e)[:1000]},
                duration_ms=(time.time() - t0) * 1000,
            )
            raise

        result_text, tokens = _extract_result(context)
        self.monitor.record_event(
            action_type="tool_call" if not is_ai else "llm_request",
            action_name=function_name,
            output_data={"text": result_text[:2000]},
            duration_ms=(time.time() - t0) * 1000,
            token_count=tokens,
            metadata={"framework": "semantic_kernel", "is_ai": is_ai},
        )
        return context


def _extract_result(context: Any) -> tuple[str, int]:
    """Pull text + token usage from the SK FunctionResult."""
    try:
        result = context.result
        text = str(getattr(result, "value", result))[:2000]
        tokens = 0
        metadata = getattr(result, "metadata", None)
        usage = metadata.get("usage", None) if isinstance(metadata, dict) else None
        if isinstance(usage, dict):
            tokens = int(usage.get("total_tokens", 0) or 0)
        elif usage is not None:
            tokens = int(getattr(usage, "total_tokens", 0) or 0)
        return text, tokens
    except Exception:
        return "", 0
