"""
DriftShield — Real-time behavioural drift detection for agentic AI systems.

Usage:
    from driftshield import DriftMonitor

    monitor = DriftMonitor(
        agent_id="my-agent",
        alert_webhook="https://hooks.slack.com/...",
    )
    agent = monitor.wrap(existing_agent)
    result = agent.invoke({"input": "do the thing"})
"""

from driftshield.models import BaselineStats, DetectorType, DriftEvent, Severity, TraceEvent
from driftshield.monitor import DriftMonitor

__version__ = "0.1.0"

__all__ = [
    "DriftMonitor",
    "TraceEvent",
    "DriftEvent",
    "BaselineStats",
    "DetectorType",
    "Severity",
]
