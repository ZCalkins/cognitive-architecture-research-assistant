"""Tests for the orchestrator skeleton.

The orchestrator is the strange-loop research subject (CLAUDE.md principle 3).
These tests pin the routing/selection/persistence cycle and — critically —
the (C) firewall encoded at the type level in the workspace.
"""

from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from src.instrumentation.probes import Probe
from src.orchestrator import Dispatcher, Orchestrator, SchemaRegistry, Workspace
from src.persistence import SQLiteStore
from src.schemas import NeuralBacking, Schema


def _schema(referent, latent_dim, signature):
    backing = NeuralBacking(nn.Linear(latent_dim, latent_dim), latent_dim)
    return Schema(referent, backing, latent_dim, context_signature=signature)


def test_registry_register_get_all_unregister():
    """Serves (C): the registry is the population lifecycle operates over."""
    registry = SchemaRegistry()
    schema = _schema("a", 4, torch.zeros(4))
    schema_id = registry.register(schema)
    assert registry.get(schema_id) is schema
    assert schema in registry.all()
    assert len(registry) == 1
    assert schema_id in registry
    registry.unregister(schema_id)
    assert registry.get(schema_id) is None
    assert len(registry) == 0


def test_registry_active_returns_sorted_by_activation():
    """Serves (A): context-relevant schemas rank first for routing."""
    registry = SchemaRegistry()
    high = _schema("high", 4, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    low = _schema("low", 4, torch.tensor([0.0, 1.0, 0.0, 0.0]))
    registry.register(high)
    registry.register(low)
    active = registry.active(torch.tensor([1.0, 0.1, 0.0, 0.0]))
    assert active[0] is high
    assert active.index(high) < active.index(low)


def test_workspace_always_includes_self_schema_state_key():
    """Serves (C) firewall AT THE TYPE LEVEL: the assembled context always has
    a ``self_schema_state`` key even when its value is None. There is nowhere
    to assemble object-level context without a reserved slot for the
    self-schema — privilege implemented architecturally, not philosophically.
    """
    workspace = Workspace(SchemaRegistry())
    context = workspace.assemble_context(torch.zeros(4), [], self_schema_state=None)
    assert "self_schema_state" in context
    assert context["self_schema_state"] is None


def test_workspace_broadcast_rolls_window():
    """Serves runtime model: the rolling window is bounded."""
    workspace = Workspace(SchemaRegistry(), rolling_window_size=3)
    for i in range(5):
        workspace.broadcast({"i": i})
    assert len(workspace._recent) == 3
    assert [content["i"] for content in workspace._recent] == [2, 3, 4]


def test_workspace_records_to_probe():
    """Serves Phase 5 criterion (D): context construction is observable."""
    workspace = Workspace(SchemaRegistry())
    probe = Probe()
    schema = _schema("a", 4, torch.ones(4))
    workspace.assemble_context(torch.ones(4), [schema], probe=probe)
    assert len(probe.get("query_context")) == 1
    assert probe.get_scalars("n_active_schemas") == [1]


def test_dispatcher_selects_top_k_by_activation():
    """Serves (B): routing selects the top-k most active schemas."""
    registry = SchemaRegistry()
    schema_1 = _schema("s1", 4, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    schema_2 = _schema("s2", 4, torch.tensor([0.0, 1.0, 0.0, 0.0]))
    schema_3 = _schema("s3", 4, torch.tensor([0.0, 0.0, 1.0, 0.0]))
    for schema in (schema_1, schema_2, schema_3):
        registry.register(schema)
    dispatcher = Dispatcher(registry, Workspace(registry))
    results = dispatcher.dispatch(torch.tensor([1.0, 0.5, 0.0, 0.0]), top_k=2)
    selected = [result[0] for result in results]
    assert len(results) == 2
    assert schema_1 in selected
    assert schema_2 in selected
    assert schema_3 not in selected


def test_dispatcher_invokes_predict_on_each_selected():
    """Serves (B): every selected schema is actually invoked."""
    registry = SchemaRegistry()
    schema_1 = _schema("s1", 4, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    schema_2 = _schema("s2", 4, torch.tensor([0.0, 1.0, 0.0, 0.0]))
    registry.register(schema_1)
    registry.register(schema_2)
    dispatcher = Dispatcher(registry, Workspace(registry))
    results = dispatcher.dispatch(torch.tensor([1.0, 1.0, 0.0, 0.0]), top_k=2)
    assert {result[0] for result in results} == {schema_1, schema_2}
    for schema, _output, alpha in results:
        assert schema.meta.activation_count == 1
        assert torch.all(alpha > 0)


def _build_orchestrator(store):
    registry = SchemaRegistry()
    registry.register(_schema("a", 4, torch.ones(4)))
    workspace = Workspace(registry)
    dispatcher = Dispatcher(registry, workspace)
    return Orchestrator(registry, workspace, dispatcher, store)


def test_orchestrator_step_records_event():
    """Serves (D): every orchestrator step is persisted as an event."""
    store = SQLiteStore(":memory:")
    orchestrator = _build_orchestrator(store)
    orchestrator.step(torch.ones(4))
    rows = store.conn.execute(
        "SELECT event_type FROM orchestrator_events WHERE event_type = 'step'"
    ).fetchall()
    assert len(rows) == 1
    store.close()


def test_orchestrator_step_increments_step_count():
    """Serves (D): the step counter is a monotonic logical clock."""
    store = SQLiteStore(":memory:")
    orchestrator = _build_orchestrator(store)
    assert orchestrator.step_count == 0
    orchestrator.step(torch.ones(4))
    orchestrator.step(torch.ones(4))
    assert orchestrator.step_count == 2
    store.close()


def test_orchestrator_session_records_open_and_close():
    """Serves Phase 4: sessions bookend the persistent self-thread."""
    store = SQLiteStore(":memory:")
    orchestrator = _build_orchestrator(store)
    with orchestrator.attach_session() as session_orchestrator:
        assert session_orchestrator is orchestrator
        orchestrator.step(torch.ones(4))
    open_rows = store.conn.execute(
        "SELECT * FROM orchestrator_events WHERE event_type = 'session_open'"
    ).fetchall()
    close_rows = store.conn.execute(
        "SELECT * FROM orchestrator_events WHERE event_type = 'session_close'"
    ).fetchall()
    assert len(open_rows) == 1
    assert len(close_rows) == 1
    store.close()


def test_registry_iter_yields_all_schemas():
    """Serves (C): the registry is iterable over its whole population."""
    registry = SchemaRegistry()
    schemas = [_schema(f"s{i}", 4, torch.zeros(4)) for i in range(3)]
    for schema in schemas:
        registry.register(schema)
    assert set(registry) == set(schemas)


def test_dispatcher_records_to_probe():
    """Serves Phase 5 criterion (D): dispatch is observable via a probe."""
    registry = SchemaRegistry()
    registry.register(_schema("a", 4, torch.ones(4)))
    dispatcher = Dispatcher(registry, Workspace(registry))
    probe = Probe()
    dispatcher.dispatch(torch.ones(4), top_k=1, probe=probe)
    assert probe.get_scalars("n_dispatched") == [1]


def test_dispatcher_uses_selector_when_provided():
    """Serves (B): an ActiveInferenceSelector drives selection when wired."""
    registry = SchemaRegistry()
    schema = _schema("a", 4, torch.ones(4))
    registry.register(schema)
    selector = MagicMock()
    selector.select.return_value = [(schema, {"total": 1.0})]
    dispatcher = Dispatcher(registry, Workspace(registry), selector=selector)
    results = dispatcher.dispatch(torch.ones(4), top_k=1)
    selector.select.assert_called_once()
    assert results[0][0] is schema


def test_dispatcher_falls_back_to_top_k_without_selector():
    """Serves Phase 5 criterion 2: the top-k baseline (ablation control) persists."""
    registry = SchemaRegistry()
    schema_1 = _schema("s1", 4, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    schema_2 = _schema("s2", 4, torch.tensor([0.0, 1.0, 0.0, 0.0]))
    registry.register(schema_1)
    registry.register(schema_2)
    dispatcher = Dispatcher(registry, Workspace(registry))
    results = dispatcher.dispatch(torch.tensor([1.0, 0.1, 0.0, 0.0]), top_k=1)
    assert len(results) == 1
    assert results[0][0] is schema_1


def test_dispatcher_records_selection_mode_to_probe():
    """Serves Phase 5 criterion 2: selection mode is recorded (1.0 AI / 0.0 baseline)."""
    registry = SchemaRegistry()
    schema = _schema("a", 4, torch.ones(4))
    registry.register(schema)

    baseline_probe = Probe()
    Dispatcher(registry, Workspace(registry)).dispatch(
        torch.ones(4), top_k=1, probe=baseline_probe
    )
    assert baseline_probe.get_scalars("selection_mode") == [0.0]

    selector = MagicMock()
    selector.select.return_value = [(schema, {"total": 1.0})]
    ai_probe = Probe()
    Dispatcher(registry, Workspace(registry), selector=selector).dispatch(
        torch.ones(4), top_k=1, probe=ai_probe
    )
    assert ai_probe.get_scalars("selection_mode") == [1.0]


def test_orchestrator_step_with_active_inference_invokes_generative_model_update():
    """Serves (B): each invocation feeds a learning update to the generative model."""
    store = SQLiteStore(":memory:")
    registry = SchemaRegistry()
    schema = _schema("a", 4, torch.ones(4))
    registry.register(schema)
    workspace = Workspace(registry)
    dispatcher = Dispatcher(registry, workspace)
    generative_model = MagicMock()
    selector = MagicMock()
    selector.select.return_value = [(schema, {"total": 1.0})]
    orchestrator = Orchestrator(
        registry, workspace, dispatcher, store,
        selector=selector, generative_model=generative_model,
    )
    orchestrator.step(torch.ones(4))
    generative_model.update.assert_called_once()
    store.close()


def test_orchestrator_raises_if_selector_without_generative_model():
    """Serves (D): a selector with no generative model is a hard config error."""
    store = SQLiteStore(":memory:")
    registry = SchemaRegistry()
    workspace = Workspace(registry)
    dispatcher = Dispatcher(registry, workspace)
    with pytest.raises(ValueError, match="requires a GenerativeModel"):
        Orchestrator(registry, workspace, dispatcher, store, selector=MagicMock())
    store.close()


def test_orchestrator_step_runs_lifecycle_when_manager_provided():
    """Serves (C): the lifecycle manager runs once per orchestrator step."""
    store = SQLiteStore(":memory:")
    registry = SchemaRegistry()
    registry.register(_schema("a", 4, torch.ones(4)))
    workspace = Workspace(registry)
    dispatcher = Dispatcher(registry, workspace)
    lifecycle_manager = MagicMock()
    lifecycle_manager.step.return_value = []
    orchestrator = Orchestrator(
        registry, workspace, dispatcher, store, lifecycle_manager=lifecycle_manager
    )
    summary = orchestrator.step(torch.ones(4))
    lifecycle_manager.step.assert_called_once()
    assert summary["n_lifecycle_events"] == 0
    store.close()
