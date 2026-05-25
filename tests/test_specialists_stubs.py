"""Tests for the Session-3 stub specialists.

The stubs only wire the pipeline (they are not load-bearing for any claim).
These tests pin the stub contract so Session 4's real specialists have a clear
behavioral baseline to replace. Serves (C) — they populate the registry the
orchestrator routes over.
"""

import torch

from src.specialists import build_v0_specialist_registry
from src.specialists.stubs import SPECIALIST_NAMES, build_novelty_vs_kb_stub


def test_stub_specialist_returns_deterministic_output():
    """Serves (D): stub output is a deterministic function of context."""
    schema = build_novelty_vs_kb_stub(8)
    context = torch.ones(8)
    output_1, alpha_1 = schema.predict(context)
    output_2, alpha_2 = schema.predict(context)
    assert torch.allclose(output_1, output_2)
    assert torch.allclose(alpha_1, alpha_2)


def test_stub_specialist_returns_valid_alpha():
    """Serves (B): stub alpha is strictly positive and correctly shaped."""
    _, alpha = build_novelty_vs_kb_stub(8).predict(torch.randn(8))
    assert torch.all(alpha > 0)
    assert alpha.shape == (8,)


def test_v0_specialist_registry_factory_creates_six_specialists():
    """Serves (C): the seed registry has the six v0 specialists."""
    assert len(build_v0_specialist_registry(latent_dim=8)) == 6


def test_v0_specialist_registry_specialists_have_correct_referents():
    """Serves (C): the six specialists carry the expected referents."""
    registry = build_v0_specialist_registry(latent_dim=8)
    assert {schema.meta.referent for schema in registry.all()} == set(SPECIALIST_NAMES)
