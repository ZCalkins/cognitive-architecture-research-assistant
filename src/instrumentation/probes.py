"""Probes — the primitive that makes architectural claims measurable.

A Probe is a lightweight observer attached to a module. It records named
tensors (or scalars) at each invocation so they can be inspected, logged, or
diffed across ablation conditions without modifying the module's forward
pass.

Strange-loop role
-----------------
The Phase 5 verification battery depends on being able to (a) record what
the self-schema and object schemas are doing, (b) compare those traces
across ablation conditions, and (c) localize causal contributions. Probes
are the substrate for all of that.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from torch import Tensor


class Probe:
    """Records named tensors during a forward pass.

    Probes are passive: they observe, they don't modify. A module that takes
    a ``probe`` argument can call ``probe.record(name, tensor)`` at any
    interesting point; the probe stores a detached CPU copy.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._traces: dict[str, list[Tensor]] = defaultdict(list)
        self._scalars: dict[str, list[float]] = defaultdict(list)

    def record(self, name: str, value: Tensor) -> None:
        """Record a tensor under ``name``. Detached and moved to CPU."""
        if not self.enabled:
            return
        self._traces[name].append(value.detach().cpu().clone())

    def record_scalar(self, name: str, value: float) -> None:
        """Record a scalar under ``name``."""
        if not self.enabled:
            return
        self._scalars[name].append(float(value))

    def get(self, name: str) -> list[Tensor]:
        return list(self._traces.get(name, []))

    def get_scalars(self, name: str) -> list[float]:
        return list(self._scalars.get(name, []))

    def clear(self) -> None:
        self._traces.clear()
        self._scalars.clear()

    def names(self) -> list[str]:
        return sorted(set(self._traces) | set(self._scalars))


class ProbeRegistry:
    """Global registry of named probes for cross-module recording."""

    def __init__(self) -> None:
        self._probes: dict[str, Probe] = {}

    def get_or_create(self, name: str, enabled: bool = True) -> Probe:
        if name not in self._probes:
            self._probes[name] = Probe(enabled=enabled)
        return self._probes[name]

    def __getitem__(self, name: str) -> Probe:
        return self._probes[name]

    def __contains__(self, name: str) -> bool:
        return name in self._probes

    def clear(self) -> None:
        for probe in self._probes.values():
            probe.clear()

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Return a dict-of-dicts view of all current probe contents."""
        out: dict[str, dict[str, Any]] = {}
        for name, probe in self._probes.items():
            out[name] = {
                "tensors": {k: probe.get(k) for k in probe._traces},
                "scalars": {k: probe.get_scalars(k) for k in probe._scalars},
            }
        return out


@contextmanager
def probing(enabled: bool = True) -> Iterator[Probe]:
    """Context manager yielding a fresh probe, auto-cleared on exit."""
    probe = Probe(enabled=enabled)
    try:
        yield probe
    finally:
        probe.clear()
