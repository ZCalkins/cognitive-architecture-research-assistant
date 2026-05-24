"""Tests for the probe instrumentation primitives.

Every test here serves Phase 5 criterion (D): instrumentation must work for
ablation traceability — if probes don't faithfully record traces, no Phase 5
ablation claim is measurable.
"""

import torch

from src.instrumentation.probes import Probe, ProbeRegistry, probing


def test_probe_records_tensors():
    """Serves Phase 5 criterion (D): traces are recorded for ablation diffs."""
    probe = Probe()
    probe.record("x", torch.tensor([1.0, 2.0]))
    recorded = probe.get("x")
    assert len(recorded) == 1
    assert torch.allclose(recorded[0], torch.tensor([1.0, 2.0]))


def test_probe_records_scalars():
    """Serves Phase 5 criterion (D): scalar metrics are recorded."""
    probe = Probe()
    probe.record_scalar("loss", 0.5)
    probe.record_scalar("loss", 0.25)
    assert probe.get_scalars("loss") == [0.5, 0.25]


def test_disabled_probe_is_noop():
    """Serves Phase 5 criterion (D): disabling a probe has zero side effects."""
    probe = Probe(enabled=False)
    probe.record("x", torch.tensor([1.0]))
    probe.record_scalar("y", 1.0)
    assert probe.get("x") == []
    assert probe.get_scalars("y") == []


def test_probe_detaches_and_moves_to_cpu():
    """Serves Phase 5 criterion (D): recorded tensors are detached, on CPU."""
    probe = Probe()
    leaf = torch.tensor([1.0, 2.0], requires_grad=True)
    probe.record("x", leaf * 2)
    recorded = probe.get("x")[0]
    assert not recorded.requires_grad
    assert recorded.device.type == "cpu"


def test_probing_context_manager_clears_on_exit():
    """Serves Phase 5 criterion (D): the probing() context auto-clears."""
    with probing() as probe:
        probe.record("x", torch.tensor([1.0]))
        assert len(probe.get("x")) == 1
    assert probe.get("x") == []


def test_registry_returns_same_probe_for_name():
    """Serves Phase 5 criterion (D): named probes are stable across lookups."""
    registry = ProbeRegistry()
    first = registry.get_or_create("a")
    second = registry.get_or_create("a")
    assert first is second


def test_registry_snapshot_collects_all():
    """Serves Phase 5 criterion (D): snapshot aggregates every probe."""
    registry = ProbeRegistry()
    registry.get_or_create("a").record_scalar("s", 1.0)
    registry.get_or_create("b").record("t", torch.tensor([1.0]))
    snapshot = registry.snapshot()
    assert set(snapshot.keys()) == {"a", "b"}
    assert snapshot["a"]["scalars"]["s"] == [1.0]
    assert len(snapshot["b"]["tensors"]["t"]) == 1
