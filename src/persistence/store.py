"""SQLite-backed state persistence for v0.

Tables mirror the five memory tracks (ARCHITECTURE.md "Memory architecture"):
schema population, lifecycle (procedural), calibration, orchestrator events
(episodic), and self-thread checkpoints (self-narrative; created now,
populated in Phase 4).

Strange-loop role
-----------------
Persistence is what makes the orchestrator a *persistent* subject rather than
a per-run process (Phase 4 identity invariants, Phase 5 criterion 5 cold
restart). The self-thread-checkpoint table is created in Session 1 even though
nothing writes to it yet, so the storage schema is stable across the phases
that will rely on it.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

    from src.schemas.base import Schema
    from src.signals.types import SignalEvent


@dataclass
class StoreEvent:
    """A typed event record."""

    timestamp: float
    event_type: str
    payload: dict


_SCHEMA_DDL = [
    """
    CREATE TABLE IF NOT EXISTS schemas (
        id TEXT PRIMARY KEY,
        referent TEXT,
        backing_type TEXT,
        parents_json TEXT,
        provenance_json TEXT,
        created_at_step INTEGER,
        last_active_step INTEGER,
        embedding_blob BLOB,
        updated_at REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lifecycle_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        schema_id TEXT,
        event_type TEXT,
        ts REAL,
        trigger_json TEXT,
        provenance_json TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS calibration_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        schema_id TEXT,
        ts REAL,
        confidence REAL,
        dirichlet_alpha_blob BLOB
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS orchestrator_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL,
        event_type TEXT,
        payload_json TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS self_thread_checkpoints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL,
        state_blob BLOB,
        identity_invariant_blob BLOB
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS engagement_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT UNIQUE,
        paper_id TEXT,
        kind TEXT,
        ts REAL,
        payload_json TEXT,
        source TEXT
    )
    """,
]


class SQLiteStore:
    """SQLite persistence for schemas, lifecycle, calibration, and events.

    Parameters
    ----------
    db_path:
        Path to the database file, or ``":memory:"`` for an in-memory store.
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        for ddl in _SCHEMA_DDL:
            self.conn.execute(ddl)
        self.conn.commit()

    def save_schema(self, schema: Schema) -> None:
        """Upsert a schema's persisted state by id."""
        parents_json = json.dumps(
            [str(p.meta.schema_id) for p in schema.meta.parents], default=str
        )
        provenance_json = json.dumps(asdict(schema.meta.provenance), default=str)
        embedding_blob = schema.embedding.detach().cpu().numpy().tobytes()
        self.conn.execute(
            """
            INSERT OR REPLACE INTO schemas (
                id, referent, backing_type, parents_json, provenance_json,
                created_at_step, last_active_step, embedding_blob, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(schema.meta.schema_id),
                schema.meta.referent,
                schema.backing.backing_type,
                parents_json,
                provenance_json,
                schema.meta.created_at_step,
                schema.meta.last_active_step,
                embedding_blob,
                time.time(),
            ),
        )
        self.conn.commit()

    def load_schema(self, schema_id: str) -> dict | None:
        """Load a schema row as a dict, or ``None`` if absent."""
        row = self.conn.execute(
            "SELECT * FROM schemas WHERE id = ?", (schema_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def record_lifecycle_event(
        self, schema_id: str, event_type: str, trigger: dict, provenance: dict
    ) -> None:
        """Record a spawn/split/merge/prune event with full provenance."""
        self.conn.execute(
            """
            INSERT INTO lifecycle_events (
                schema_id, event_type, ts, trigger_json, provenance_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(schema_id),
                event_type,
                time.time(),
                json.dumps(trigger, default=str),
                json.dumps(provenance, default=str),
            ),
        )
        self.conn.commit()

    def record_calibration(
        self, schema_id: str, confidence: float, dirichlet_alpha: Tensor
    ) -> None:
        """Record a per-schema confidence + Dirichlet alpha snapshot."""
        alpha_blob = dirichlet_alpha.detach().cpu().numpy().tobytes()
        self.conn.execute(
            """
            INSERT INTO calibration_history (
                schema_id, ts, confidence, dirichlet_alpha_blob
            ) VALUES (?, ?, ?, ?)
            """,
            (str(schema_id), time.time(), float(confidence), alpha_blob),
        )
        self.conn.commit()

    def record_orchestrator_event(self, event_type: str, payload: dict) -> None:
        """Record an orchestrator-level event (step, session open/close)."""
        self.conn.execute(
            """
            INSERT INTO orchestrator_events (ts, event_type, payload_json)
            VALUES (?, ?, ?)
            """,
            (time.time(), event_type, json.dumps(payload, default=str)),
        )
        self.conn.commit()

    def record_engagement_signal(self, event: SignalEvent) -> None:
        """Persist an engagement SignalEvent (claim (A) substrate)."""
        self.conn.execute(
            """
            INSERT INTO engagement_signals (
                event_id, paper_id, kind, ts, payload_json, source
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(event.event_id),
                event.paper_id,
                event.kind.value,
                event.timestamp.timestamp(),
                json.dumps(event.payload, default=str),
                event.source,
            ),
        )
        self.conn.commit()

    def engagement_signals_for_paper(self, paper_id: str) -> list[dict]:
        """All engagement signals for a paper, oldest first."""
        rows = self.conn.execute(
            "SELECT * FROM engagement_signals WHERE paper_id = ? ORDER BY ts ASC",
            (paper_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def recent_engagement_signals(self, limit: int = 100) -> list[dict]:
        """The most recent engagement signals, newest first."""
        rows = self.conn.execute(
            "SELECT * FROM engagement_signals ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self.conn.close()
