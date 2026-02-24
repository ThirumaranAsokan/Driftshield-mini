"""Core data models for DriftShield."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MED"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @classmethod
    def from_score(cls, score: float) -> Severity:
        if score >= 0.9:
            return cls.CRITICAL
        elif score >= 0.7:
            return cls.HIGH
        elif score >= 0.5:
            return cls.MEDIUM
        return cls.LOW


class DetectorType(str, Enum):
    ACTION_LOOP = "action_loop"
    GOAL_DRIFT = "goal_drift"
    RESOURCE_SPIKE = "resource_spike"


@dataclass
class TraceEvent:
    """A single captured event from an agent run."""

    agent_id: str
    run_id: str
    action_type: str  # "tool_call", "llm_request", "state_transition"
    action_name: str  # e.g. "search_database", "format_result"
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    token_count: int = 0
    input_data: dict[str, Any] = field(default_factory=dict)
    output_data: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "agent_id": self.agent_id,
            "run_id": self.run_id,
            "action_type": self.action_type,
            "action_name": self.action_name,
            "timestamp": self.timestamp,
            "token_count": self.token_count,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceEvent:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class DriftEvent:
    """A detected drift incident."""

    agent_id: str
    run_id: str
    detector: DetectorType
    severity: Severity
    score: float  # 0.0 - 1.0
    message: str
    suggested_action: str
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "agent_id": self.agent_id,
            "run_id": self.run_id,
            "detector": self.detector.value,
            "severity": self.severity.value,
            "score": self.score,
            "message": self.message,
            "suggested_action": self.suggested_action,
            "timestamp": self.timestamp,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DriftEvent:
        data = dict(data)
        if isinstance(data.get("detector"), str):
            data["detector"] = DetectorType(data["detector"])
        if isinstance(data.get("severity"), str):
            data["severity"] = Severity(data["severity"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RunSummary:
    """Summary statistics for a single agent run."""

    agent_id: str
    run_id: str
    start_time: float
    end_time: float
    total_tokens: int = 0
    total_tool_calls: int = 0
    total_llm_calls: int = 0
    action_sequence: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    drift_events: list[DriftEvent] = field(default_factory=list)
    output_text: str = ""


@dataclass
class BaselineStats:
    """Baseline statistics built during calibration."""

    agent_id: str
    calibration_runs: int = 0
    mean_tokens_per_run: float = 0.0
    std_tokens_per_run: float = 0.0
    mean_tools_per_run: float = 0.0
    std_tools_per_run: float = 0.0
    mean_duration_ms: float = 0.0
    std_duration_ms: float = 0.0
    common_sequences: list[list[str]] = field(default_factory=list)
    goal_embedding: list[float] | None = None
    mean_goal_similarity: float = 0.0
    std_goal_similarity: float = 0.0
    is_calibrated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "calibration_runs": self.calibration_runs,
            "mean_tokens_per_run": self.mean_tokens_per_run,
            "std_tokens_per_run": self.std_tokens_per_run,
            "mean_tools_per_run": self.mean_tools_per_run,
            "std_tools_per_run": self.std_tools_per_run,
            "mean_duration_ms": self.mean_duration_ms,
            "std_duration_ms": self.std_duration_ms,
            "common_sequences": self.common_sequences,
            "goal_embedding": self.goal_embedding,
            "mean_goal_similarity": self.mean_goal_similarity,
            "std_goal_similarity": self.std_goal_similarity,
            "is_calibrated": self.is_calibrated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaselineStats:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
