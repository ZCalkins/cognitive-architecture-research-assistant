"""Instrumentation primitives: probes, ablation harnesses, metrics.

The falsifiability principle (Principle 1 in CLAUDE.md) requires that every
architectural claim be measurable. This subpackage provides the primitives
that make that possible — probes attached to modules, harnesses for ablation
studies, and the Phase 5 verification metrics.
"""

from src.instrumentation.probes import Probe, ProbeRegistry, probing

__all__ = ["Probe", "ProbeRegistry", "probing"]
