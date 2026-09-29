"""SQLite-based local storage for traces, drift events, and baselines."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from driftshield.models import BaselineStats, DriftEvent, TraceEvent

DEFAULT_DB_PATH = Path.home() / ".driftshield" / "driftshield.db"


class TraceStore:
    """Thread-safe SQLite store for all DriftShield data."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    @property
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path))
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn

    def _init_db(self) -> None:
        conn = self._conn
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS trace_events (
                event_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                action_type TEXT NOT NULL,
                action_name TEXT NOT NULL,
                timestamp REAL NOT NULL,
                token_count INTEGER DEFAULT 0,
                input_data TEXT DEFAULT '{}',
                output_data TEXT DEFAULT '{}',
                duration_ms REAL DEFAULT 0.0,
                metadata TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS drift_events (
                event_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                detector TEXT NOT NULL,
                severity TEXT NOT NULL,
                score REAL NOT NULL,
                message TEXT NOT NULL,
                suggested_action TEXT NOT NULL,
                timestamp REAL NOT NULL,
                context TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS baselines (
                agent_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_traces_agent ON trace_events(agent_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_traces_run ON trace_events(run_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_drift_agent ON drift_events(agent_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_drift_severity ON drift_events(severity, timestamp);
        """)
        conn.commit()

    # ── Trace Events ──────────────────────────────────────────────

    def save_trace(self, event: TraceEvent) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO trace_events
               (event_id, agent_id, run_id, action_type, action_name,
                timestamp, token_count, input_data, output_data, duration_ms, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.event_id,
                event.agent_id,
                event.run_id,
                event.action_type,
                event.action_name,
                event.timestamp,
                event.token_count,
                json.dumps(event.input_data),
                json.dumps(event.output_data),
                event.duration_ms,
                json.dumps(event.metadata),
            ),
        )
        self._conn.commit()

    def get_traces(
        self,
        agent_id: str | None = None,
        run_id: str | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[TraceEvent]:
        query = "SELECT * FROM trace_events WHERE 1=1"
        params: list[Any] = []

        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        if since:
            query += " AND timestamp >= ?"
            params.append(since)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_trace(row) for row in rows]

    def get_run_traces(self, agent_id: str, run_id: str) -> list[TraceEvent]:
        """Get all traces for a specific run, ordered chronologically."""
        rows = self._conn.execute(
            "SELECT * FROM trace_events WHERE agent_id = ? AND run_id = ? ORDER BY timestamp ASC",
            (agent_id, run_id),
        ).fetchall()
        return [self._row_to_trace(row) for row in rows]

    def get_recent_actions(self, agent_id: str, run_id: str, window: int = 20) -> list[str]:
        """Get the most recent action names for loop detection."""
        rows = self._conn.execute(
            """SELECT action_name FROM trace_events
               WHERE agent_id = ? AND run_id = ? AND action_type = 'tool_call'
               ORDER BY timestamp DESC LIMIT ?""",
            (agent_id, run_id, window),
        ).fetchall()
        return [row["action_name"] for row in reversed(rows)]

    def get_run_ids(self, agent_id: str, limit: int = 50) -> list[str]:
        """Get distinct run IDs for an agent, most recent first."""
        rows = self._conn.execute(
            """SELECT DISTINCT run_id FROM trace_events
               WHERE agent_id = ? ORDER BY timestamp DESC LIMIT ?""",
            (agent_id, limit),
        ).fetchall()
        return [row["run_id"] for row in rows]

    # ── Drift Events ──────────────────────────────────────────────

    def save_drift(self, event: DriftEvent) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO drift_events
               (event_id, agent_id, run_id, detector, severity, score,
                message, suggested_action, timestamp, context)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.event_id,
                event.agent_id,
                event.run_id,
                event.detector.value,
                event.severity.value,
                event.score,
                event.message,
                event.suggested_action,
                event.timestamp,
                json.dumps(event.context),
            ),
        )
        self._conn.commit()

    def get_drift_events(
        self,
        agent_id: str | None = None,
        since: float | None = None,
        severity: str | None = None,
        limit: int = 50,
    ) -> list[DriftEvent]:
        query = "SELECT * FROM drift_events WHERE 1=1"
        params: list[Any] = []

        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)
        if since:
            query += " AND timestamp >= ?"
            params.append(since)
        if severity:
            query += " AND severity = ?"
            params.append(severity)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_drift(row) for row in rows]

    # ── Baselines ─────────────────────────────────────────────────

    def save_baseline(self, baseline: BaselineStats) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO baselines (agent_id, data, updated_at) VALUES (?, ?, ?)",
            (baseline.agent_id, json.dumps(baseline.to_dict()), time.time()),
        )
        self._conn.commit()

    def get_baseline(self, agent_id: str) -> BaselineStats | None:
        row = self._conn.execute(
            "SELECT data FROM baselines WHERE agent_id = ?", (agent_id,)
        ).fetchone()
        if row:
            return BaselineStats.from_dict(json.loads(row["data"]))
        return None

    # ── Run Stats (for baseline building) ─────────────────────────

    def get_run_stats(self, agent_id: str, run_id: str) -> dict[str, Any]:
        """Get aggregate stats for a single run."""
        row = self._conn.execute(
            """SELECT
                COUNT(*) as event_count,
                SUM(token_count) as total_tokens,
                SUM(CASE WHEN action_type = 'tool_call' THEN 1 ELSE 0 END) as tool_calls,
                SUM(CASE WHEN action_type = 'llm_request' THEN 1 ELSE 0 END) as llm_calls,
                MIN(timestamp) as start_time,
                MAX(timestamp) as end_time,
                SUM(duration_ms) as total_duration_ms
            FROM trace_events WHERE agent_id = ? AND run_id = ?""",
            (agent_id, run_id),
        ).fetchone()

        return {
            "event_count": row["event_count"] or 0,
            "total_tokens": row["total_tokens"] or 0,
            "tool_calls": row["tool_calls"] or 0,
            "llm_calls": row["llm_calls"] or 0,
            "start_time": row["start_time"] or 0,
            "end_time": row["end_time"] or 0,
            "total_duration_ms": row["total_duration_ms"] or 0,
        }

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _row_to_trace(row: sqlite3.Row) -> TraceEvent:
        return TraceEvent(
            event_id=row["event_id"],
            agent_id=row["agent_id"],
            run_id=row["run_id"],
            action_type=row["action_type"],
            action_name=row["action_name"],
            timestamp=row["timestamp"],
            token_count=row["token_count"],
            input_data=json.loads(row["input_data"]),
            output_data=json.loads(row["output_data"]),
            duration_ms=row["duration_ms"],
            metadata=json.loads(row["metadata"]),
        )

    @staticmethod
    def _row_to_drift(row: sqlite3.Row) -> DriftEvent:
        return DriftEvent(
            event_id=row["event_id"],
            agent_id=row["agent_id"],
            run_id=row["run_id"],
            detector=row["detector"],
            severity=row["severity"],
            score=row["score"],
            message=row["message"],
            suggested_action=row["suggested_action"],
            timestamp=row["timestamp"],
            context=json.loads(row["context"]),
        )

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
