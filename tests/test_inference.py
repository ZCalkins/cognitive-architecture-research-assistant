"""Tests for active-inference action selection (Session 2).

Active inference is the orchestrator's action-selection mechanism — the
load-bearing target for Phase 5 criterion 2 (downward causation traceability):
in Phase 3 the self-schema modulates the GenerativeModel via the hypernetwork.
These tests pin the Session 2 standalone behavior of that mechanism.
"""

import torch
from torch import nn

from src.inference import (
    ActiveInferenceSelector,
    ConfidencePreferences,
    GenerativeModel,
    UniformPreferences,
)
from src.instrumentation.probes import Probe
from src.schemas import NeuralBacking, Schema


def _schema(referent, latent_dim, signature):
    backing = NeuralBacking(nn.Linear(latent_dim, latent_dim), latent_dim)
    return Schema(referent, backing, latent_dim, context_signature=signature)


def test_generative_model_predict_shapes():
    """Serves (B): the generative model predicts output/alpha/uncertainty."""
    latent_dim = 4
    model = GenerativeModel(latent_dim=latent_dim)
    schema = _schema("a", latent_dim, torch.randn(latent_dim))
    output, alpha, uncertainty = model.predict(schema, torch.randn(latent_dim))
    assert output.shape == (latent_dim,)
    assert alpha.shape == (latent_dim,)
    assert torch.all(alpha > 0)
    assert uncertainty.dim() == 0


def test_generative_model_update_reduces_loss():
    """Serves (B): the generative model learns from observations."""
    torch.manual_seed(0)
    latent_dim = 3
    model = GenerativeModel(latent_dim=latent_dim)
    schema = _schema("a", latent_dim, torch.ones(latent_dim))
    context = torch.ones(latent_dim)
    target_output = torch.full((latent_dim,), 2.0)
    target_alpha = torch.full((latent_dim,), 1.5)
    first_loss = model.update(schema, context, target_output, target_alpha)
    last_loss = first_loss
    for _ in range(50):
        last_loss = model.update(schema, context, target_output, target_alpha)
    assert last_loss < first_loss


def test_confidence_preferences_higher_alpha_higher_score():
    """Serves (A): higher confidence is more preferred (pre-engagement default)."""
    preferences = ConfidencePreferences()
    low = preferences.score(torch.zeros(2), torch.tensor([1.0, 1.0]))
    high = preferences.score(torch.zeros(2), torch.tensor([3.0, 3.0]))
    assert high > low


def test_uniform_preferences_constant_zero():
    """Serves Phase 5 criterion 2: the pragmatic-ablation baseline is constant."""
    preferences = UniformPreferences()
    assert preferences.score(torch.randn(5), torch.rand(5) + 0.1) == 0.0


def test_active_inference_selector_returns_top_k():
    """Serves (B): the selector returns exactly k specialists."""
    torch.manual_seed(0)
    latent_dim = 4
    selector = ActiveInferenceSelector(GenerativeModel(latent_dim=latent_dim), ConfidencePreferences())
    schemas = [_schema(f"s{i}", latent_dim, torch.randn(latent_dim)) for i in range(4)]
    selected = selector.select(schemas, torch.randn(latent_dim), k=2)
    assert len(selected) == 2
    assert all(isinstance(schema, Schema) for schema, _ in selected)


def test_active_inference_selector_score_action_returns_required_fields():
    """Serves Phase 5 criterion (D): every score is fully inspectable."""
    latent_dim = 3
    selector = ActiveInferenceSelector(GenerativeModel(latent_dim=latent_dim), ConfidencePreferences())
    schema = _schema("a", latent_dim, torch.randn(latent_dim))
    score = selector.score_action(schema, torch.randn(latent_dim))
    for key in (
        "schema_id",
        "pragmatic",
        "epistemic",
        "total",
        "predicted_output",
        "predicted_alpha",
        "predicted_uncertainty",
    ):
        assert key in score


def test_active_inference_epistemic_weight_zero_collapses_to_pragmatic_only():
    """Serves Phase 5 criterion 2: ablating epistemic value leaves pragmatic ranking."""
    torch.manual_seed(0)
    latent_dim = 4
    context = torch.randn(latent_dim)
    selector = ActiveInferenceSelector(
        GenerativeModel(latent_dim=latent_dim),
        ConfidencePreferences(),
        epistemic_weight=0.0,
        pragmatic_weight=1.0,
    )
    schemas = [_schema(f"s{i}", latent_dim, torch.randn(latent_dim)) for i in range(5)]
    selected_order = [schema.meta.schema_id for schema, _ in selector.select(schemas, context, k=5)]
    pragmatic_order = [
        schema.meta.schema_id
        for schema, _ in sorted(
            ((schema, selector.score_action(schema, context)) for schema in schemas),
            key=lambda pair: pair[1]["pragmatic"],
            reverse=True,
        )
    ]
    assert selected_order == pragmatic_order


def test_active_inference_pragmatic_weight_zero_collapses_to_epistemic_only():
    """Serves Phase 5 criterion 2: ablating pragmatic value leaves epistemic ranking."""
    torch.manual_seed(0)
    latent_dim = 4
    context = torch.randn(latent_dim)
    selector = ActiveInferenceSelector(
        GenerativeModel(latent_dim=latent_dim),
        ConfidencePreferences(),
        epistemic_weight=1.0,
        pragmatic_weight=0.0,
    )
    schemas = [_schema(f"s{i}", latent_dim, torch.randn(latent_dim)) for i in range(5)]
    selected_order = [schema.meta.schema_id for schema, _ in selector.select(schemas, context, k=5)]
    epistemic_order = [
        schema.meta.schema_id
        for schema, _ in sorted(
            ((schema, selector.score_action(schema, context)) for schema in schemas),
            key=lambda pair: pair[1]["epistemic"],
            reverse=True,
        )
    ]
    assert selected_order == epistemic_order


def test_active_inference_selector_records_to_probe():
    """Serves Phase 5 criterion (D): per-schema scores are probed."""
    latent_dim = 3
    selector = ActiveInferenceSelector(GenerativeModel(latent_dim=latent_dim), ConfidencePreferences())
    schema = _schema("a", latent_dim, torch.randn(latent_dim))
    probe = Probe()
    selector.select([schema], torch.randn(latent_dim), k=1, probe=probe)
    short = str(schema.meta.schema_id)[:8]
    assert probe.get_scalars(f"pragmatic:{short}")
    assert probe.get_scalars(f"epistemic:{short}")
    assert probe.get_scalars(f"total:{short}")
