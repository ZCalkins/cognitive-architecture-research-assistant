"""Tests for the SQLite persistence store.

Persistence is what makes the orchestrator a persistent subject across
sessions and substrate changes (Phase 4 identity invariants; Phase 5
criterion 5 cold restart). These tests pin the storage round-trips.
"""

import json
from datetime import datetime

import numpy as np
import pytest
import torch
from torch import nn

from src.persistence import SQLiteStore
from src.schemas import NeuralBacking, Schema
from src.signals import SignalEvent, SignalKind


def _schema(referent="a", latent_dim=4):
    backing = NeuralBacking(nn.Linear(latent_dim, latent_dim), latent_dim)
    return Schema(referent, backing, latent_dim)


def test_in_memory_store_creates_tables():
    """Serves (D): the full five-track storage schema exists from Session 1."""
    store = SQLiteStore(":memory:")
    rows = store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    names = {row["name"] for row in rows}
    expected = {
        "schemas",
        "lifecycle_events",
        "calibration_history",
        "orchestrator_events",
        "self_thread_checkpoints",
    }
    assert expected.issubset(names)
    store.close()


def test_save_load_schema_round_trip():
    """Serves (A): a schema's persisted identity round-trips."""
    store = SQLiteStore(":memory:")
    schema = _schema("novelty-vs-KB", 4)
    store.save_schema(schema)
    loaded = store.load_schema(str(schema.meta.schema_id))
    assert loaded is not None
    assert loaded["referent"] == "novelty-vs-KB"
    assert loaded["backing_type"] == "neural"
    store.close()


def test_save_schema_upsert():
    """Serves (A): re-saving a schema updates rather than duplicates."""
    store = SQLiteStore(":memory:")
    schema = _schema("a", 4)
    store.save_schema(schema)
    schema.meta.last_active_step = 7
    store.save_schema(schema)
    count = store.conn.execute(
        "SELECT COUNT(*) AS c FROM schemas WHERE id = ?", (str(schema.meta.schema_id),)
    ).fetchone()["c"]
    assert count == 1
    loaded = store.load_schema(str(schema.meta.schema_id))
    assert loaded["last_active_step"] == 7
    store.close()


def test_record_lifecycle_event():
    """Serves (D): lifecycle events are persisted with their triggers."""
    store = SQLiteStore(":memory:")
    store.record_lifecycle_event(
        "sid-1", "spawn", {"reason": "prediction error"}, {"parent_ids": []}
    )
    rows = store.conn.execute(
        "SELECT * FROM lifecycle_events WHERE schema_id = 'sid-1'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["event_type"] == "spawn"
    assert json.loads(rows[0]["trigger_json"])["reason"] == "prediction error"
    store.close()


def test_record_calibration_with_tensor_blob():
    """Serves calibration track: the Dirichlet alpha blob round-trips."""
    store = SQLiteStore(":memory:")
    alpha = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32)
    store.record_calibration("sid-1", 0.8, alpha)
    row = store.conn.execute(
        "SELECT * FROM calibration_history WHERE schema_id = 'sid-1'"
    ).fetchone()
    assert row["confidence"] == pytest.approx(0.8)
    restored = np.frombuffer(row["dirichlet_alpha_blob"], dtype=np.float32)
    assert np.allclose(restored, np.array([1.5, 2.5, 3.5], dtype=np.float32))
    store.close()


def test_record_orchestrator_event_with_json_payload():
    """Serves (D): orchestrator event payloads round-trip through JSON."""
    store = SQLiteStore(":memory:")
    payload = {"step": 3, "schema_ids": ["a", "b"], "mean_alpha_sum": 1.25}
    store.record_orchestrator_event("step", payload)
    row = store.conn.execute(
        "SELECT * FROM orchestrator_events WHERE event_type = 'step'"
    ).fetchone()
    assert json.loads(row["payload_json"]) == payload
    store.close()


def test_load_missing_schema_returns_none():
    """Serves (A): loading an unknown schema id returns None, not an error."""
    store = SQLiteStore(":memory:")
    assert store.load_schema("does-not-exist") is None
    store.close()


def test_engagement_signals_table_created():
    """Serves (A): the engagement_signals table exists from Session 3."""
    store = SQLiteStore(":memory:")
    rows = store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    assert "engagement_signals" in {row["name"] for row in rows}
    store.close()


def test_record_engagement_signal_round_trip():
    """Serves (A): a signal event round-trips through the store."""
    store = SQLiteStore(":memory:")
    event = SignalEvent(
        paper_id="p1",
        kind=SignalKind.READ,
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        payload={"x": 1},
        source="cli",
    )
    store.record_engagement_signal(event)
    rows = store.engagement_signals_for_paper("p1")
    assert len(rows) == 1
    assert rows[0]["kind"] == "read"
    assert rows[0]["source"] == "cli"
    assert json.loads(rows[0]["payload_json"]) == {"x": 1}
    store.close()


def test_engagement_signals_for_paper_returns_ordered_list():
    """Serves (A): a paper's signals come back oldest-first."""
    store = SQLiteStore(":memory:")
    store.record_engagement_signal(
        SignalEvent(paper_id="p", kind=SignalKind.OPENED, timestamp=datetime(2024, 1, 1, 1))
    )
    store.record_engagement_signal(
        SignalEvent(paper_id="p", kind=SignalKind.READ, timestamp=datetime(2024, 1, 1, 2))
    )
    rows = store.engagement_signals_for_paper("p")
    assert [row["kind"] for row in rows] == ["opened", "read"]
    store.close()


def test_recent_engagement_signals_limit_and_order():
    """Serves (A): recent signals are newest-first and limited."""
    store = SQLiteStore(":memory:")
    for hour in range(5):
        store.record_engagement_signal(
            SignalEvent(paper_id=f"p{hour}", kind=SignalKind.FLAG, timestamp=datetime(2024, 1, 1, hour))
        )
    rows = store.recent_engagement_signals(limit=3)
    assert len(rows) == 3
    assert rows[0]["paper_id"] == "p4"
    store.close()
