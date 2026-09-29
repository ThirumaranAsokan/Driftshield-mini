"""Audit log export — CSV and JSON for compliance officers (FCA / EU AI Act).

Usage (CLI):
    driftshield export --agent my-agent --output audit.csv
    driftshield export --agent my-agent --output audit.json --format json
    driftshield export --agent my-agent --output drift.json --drift-only
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from driftshield_mini.models import DriftEvent, TraceEvent
from driftshield_mini.storage import TraceStore

TRACE_FIELDS = [
    "event_id", "agent_id", "run_id", "action_type", "action_name",
    "timestamp", "token_count", "input_data", "output_data",
    "duration_ms", "metadata",
]

DRIFT_FIELDS = [
    "event_id", "agent_id", "run_id", "detector", "severity", "score",
    "message", "suggested_action", "timestamp", "context",
]


def _flatten(event_dict: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    row = {}
    for f in fields:
        v = event_dict.get(f, "")
        if isinstance(v, (dict, list)):
            v = json.dumps(v)
        row[f] = v
    return row


def export_traces(
    store: TraceStore,
    output: str | Path,
    agent_id: str | None = None,
    fmt: str = "csv",
    hours: float | None = None,
    limit: int = 100000,
) -> Path:
    """Export trace events to CSV or JSON."""
    since = time.time() - hours * 3600 if hours else None
    events = store.get_traces(agent_id=agent_id, since=since, limit=limit)
    return _write(events, TRACE_FIELDS, output, fmt)


def export_drift_events(
    store: TraceStore,
    output: str | Path,
    agent_id: str | None = None,
    fmt: str = "csv",
    hours: float | None = None,
    limit: int = 100000,
) -> Path:
    """Export drift (incident) events to CSV or JSON."""
    since = time.time() - hours * 3600 if hours else None
    events = store.get_drift_events(agent_id=agent_id, since=since, limit=limit)
    return _write(events, DRIFT_FIELDS, output, fmt)


def _write(events, fields: list[str], output: str | Path, fmt: str) -> Path:
    output = Path(output)
    fmt = fmt.lower().lstrip(".")
    if fmt not in ("csv", "json"):
        raise ValueError(f"Unsupported format '{fmt}'. Use csv or json.")

    rows = [_flatten(e.to_dict(), fields) for e in events]

    if fmt == "json":
        output.write_text(json.dumps(rows, indent=2, default=str))
    else:
        with output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    return output
