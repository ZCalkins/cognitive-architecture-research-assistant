"""Stub specialists — pipeline plumbing ONLY (Session 3).

These are deterministic-from-context stubs: NOT learned, NOT real specialists.
They exist solely to exercise the end-to-end ingestion -> triage ->
engagement-signal pipeline so the (A) signal flow can be developed before the
real specialists exist.

Session 4 replaces every one of these with a real ``LLMBacking``-backed
``Schema`` carrying a concrete prompt template and a structured-output
protocol. Do not extend the stubs; do not let stub behavior leak into the
Session 4 implementations.
"""

from __future__ import annotations

import hashlib

import torch
from torch import Tensor

from src.orchestrator import SchemaRegistry
from src.schemas import Schema, SymbolicBacking

SPECIALIST_NAMES = (
    "novelty_vs_kb",
    "methodological_rigor",
    "theoretical_claim_evaluator",
    "relevance_to_projects",
    "citation_graph_position",
    "author_history",
)


def _make_stub_rule(referent: str, latent_dim: int):
    """Build a deterministic-from-context rule (uses a local RNG, not global)."""

    def rule(context: Tensor, slot_values: dict | None) -> tuple[Tensor, Tensor]:
        context_bytes = context.detach().cpu().numpy().tobytes()
        digest = hashlib.sha1(referent.encode("utf-8") + context_bytes).digest()[:4]
        seed = int.from_bytes(digest, "big")
        generator = torch.Generator().manual_seed(seed)
        output = torch.randn(latent_dim, generator=generator)
        alpha = torch.ones(latent_dim) + output.abs()
        return output, alpha

    return rule


def _build_stub(referent: str, latent_dim: int) -> Schema:
    return Schema(
        referent=referent,
        backing=SymbolicBacking(rule=_make_stub_rule(referent, latent_dim)),
        latent_dim=latent_dim,
    )


def build_novelty_vs_kb_stub(latent_dim: int) -> Schema:
    return _build_stub("novelty_vs_kb", latent_dim)


def build_methodological_rigor_stub(latent_dim: int) -> Schema:
    return _build_stub("methodological_rigor", latent_dim)


def build_theoretical_claim_evaluator_stub(latent_dim: int) -> Schema:
    return _build_stub("theoretical_claim_evaluator", latent_dim)


def build_relevance_to_projects_stub(latent_dim: int) -> Schema:
    return _build_stub("relevance_to_projects", latent_dim)


def build_citation_graph_position_stub(latent_dim: int) -> Schema:
    return _build_stub("citation_graph_position", latent_dim)


def build_author_history_stub(latent_dim: int) -> Schema:
    return _build_stub("author_history", latent_dim)


def build_v0_specialist_registry(latent_dim: int = 1024) -> SchemaRegistry:
    """Wire the six stub specialists into a registry for the orchestrator."""
    registry = SchemaRegistry()
    for name in SPECIALIST_NAMES:
        registry.register(_build_stub(name, latent_dim))
    return registry
