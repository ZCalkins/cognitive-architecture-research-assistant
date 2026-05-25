"""Tests for the Piagetian lifecycle API (Session 2 scaffolding).

These serve (C) self-organizing specialist emergence. Session 2 wires the
manager API and the compose/prune operations; trigger policies are Phase 2,
so the trigger stubs are asserted to propose no events yet.
"""

import pytest
import torch
from torch import nn

from src.orchestrator import SchemaRegistry
from src.persistence import SQLiteStore
from src.schemas import NeuralBacking, Schema
from src.schemas.lifecycle import (
    LifecycleEvent,
    LifecycleManager,
    LifecycleTrigger,
    MergeTrigger,
    PruneTrigger,
    SpawnTrigger,
    SplitTrigger,
)


def _schema(referent="a", latent_dim=4):
    backing = NeuralBacking(nn.Linear(latent_dim, latent_dim), latent_dim)
    return Schema(referent, backing, latent_dim)


class _FakeTrigger(LifecycleTrigger):
    trigger_type = "fake"

    def __init__(self, events):
        self._events = events

    def evaluate(self, schemas, context):
        return list(self._events)


def test_lifecycle_event_dataclass():
    """Serves (C): events carry provenance; metadata defaults to {}."""
    event = LifecycleEvent(
        event_type="spawn", affected_schema_ids=[], trigger_description="x", timestamp=1.0
    )
    assert event.metadata == {}


def test_spawn_trigger_returns_empty_in_session2():
    """Session 2 stub; Phase 2 will implement real spawn dynamics."""
    assert SpawnTrigger().evaluate([_schema()], torch.zeros(4)) == []


def test_split_trigger_returns_empty_in_session2():
    """Session 2 stub; Phase 2 will implement real split dynamics."""
    assert SplitTrigger().evaluate([_schema()], torch.zeros(4)) == []


def test_merge_trigger_returns_empty_in_session2():
    """Session 2 stub; Phase 2 will implement real merge dynamics."""
    assert MergeTrigger().evaluate([_schema()], torch.zeros(4)) == []


def test_prune_trigger_returns_empty_in_session2():
    """Session 2 stub; Phase 2 will implement real prune dynamics."""
    assert PruneTrigger().evaluate([_schema()], torch.zeros(4)) == []


def test_lifecycle_manager_evaluate_returns_concatenated_events():
    """Serves (C): the manager collects events from all triggers."""
    store = SQLiteStore(":memory:")
    event = LifecycleEvent("prune", [], "fake", 0.0)
    manager = LifecycleManager([_FakeTrigger([event])], SchemaRegistry(), store)
    assert manager.evaluate(torch.zeros(4)) == [event]
    store.close()


def test_lifecycle_manager_apply_prune_unregisters():
    """Serves (C): applying a prune removes the schema from the population."""
    store = SQLiteStore(":memory:")
    registry = SchemaRegistry()
    schema_id = registry.register(_schema())
    manager = LifecycleManager([], registry, store)
    manager.apply([LifecycleEvent("prune", [schema_id], "idle", 0.0)])
    assert schema_id not in registry
    store.close()


def test_lifecycle_manager_apply_compose_creates_composite():
    """Serves combinatorial novelty: a compose event registers a composite."""
    store = SQLiteStore(":memory:")
    registry = SchemaRegistry()
    id_a = registry.register(_schema("a"))
    id_b = registry.register(_schema("b"))
    manager = LifecycleManager([], registry, store)
    manager.apply(
        [LifecycleEvent("compose", [id_a, id_b], "compose", 0.0, {"mode": "sequential"})]
    )
    assert len(registry) == 3
    store.close()


def test_lifecycle_manager_apply_spawn_raises_phase2():
    """Serves (D): spawn dynamics are honestly deferred to Phase 2."""
    store = SQLiteStore(":memory:")
    manager = LifecycleManager([], SchemaRegistry(), store)
    with pytest.raises(NotImplementedError, match="Phase 2"):
        manager.apply([LifecycleEvent("spawn", [], "x", 0.0)])
    store.close()


def test_lifecycle_manager_apply_records_to_store():
    """Serves (D): applied lifecycle events are persisted."""
    store = SQLiteStore(":memory:")
    registry = SchemaRegistry()
    schema_id = registry.register(_schema())
    manager = LifecycleManager([], registry, store)
    manager.apply([LifecycleEvent("prune", [schema_id], "idle", 0.0)])
    rows = store.conn.execute(
        "SELECT * FROM lifecycle_events WHERE event_type = 'prune'"
    ).fetchall()
    assert len(rows) == 1
    store.close()


def test_lifecycle_manager_step_evaluates_and_applies():
    """Serves (C): step evaluates triggers and applies the proposed events."""
    store = SQLiteStore(":memory:")
    registry = SchemaRegistry()
    id_a = registry.register(_schema("a"))
    id_b = registry.register(_schema("b"))
    compose_event = LifecycleEvent("compose", [id_a, id_b], "compose", 0.0, {"mode": "parallel"})
    manager = LifecycleManager([_FakeTrigger([compose_event])], registry, store)
    events = manager.step(torch.zeros(4))
    assert len(events) == 1
    assert len(registry) == 3
    store.close()
