"""Tests for the polymorphic schema framework.

These exercise the universal schema interface that the strange loop closes
over: polymorphic backings (substrate for (C) self-organizing specialists and
(B) typed downward causation), provenance and lifecycle operators (Phase 2),
analogical mapping (combinatorial novelty), and ablation hooks (Phase 5
criterion 2).
"""

import pytest
import torch
from torch import nn

from src.schemas import (
    CompositeBacking,
    LLMBacking,
    NeuralBacking,
    Schema,
    SymbolicBacking,
    TypedSlot,
)


def _neural_schema(referent="s", latent_dim=4, signature=None):
    module = nn.Linear(latent_dim, latent_dim)
    backing = NeuralBacking(module, latent_dim)
    return Schema(referent, backing, latent_dim, context_signature=signature)


def test_neural_backing_predict_shapes():
    """Serves (C)/(B): neural backing yields a strictly positive Dirichlet."""
    latent_dim = 4
    backing = NeuralBacking(nn.Linear(latent_dim, latent_dim), latent_dim)
    output, alpha = backing.forward(torch.randn(2, latent_dim))
    assert output.shape == (2, latent_dim)
    assert alpha.shape == (2, latent_dim)
    assert torch.all(alpha > 0)


def test_llm_backing_raises_without_provider():
    """Serves (C) firewall: LLMBacking cannot run without a wired provider."""
    backing = LLMBacking(prompt_template="{paper}")
    with pytest.raises(NotImplementedError, match="requires an LLMProvider"):
        backing.forward(torch.zeros(1), slot_values=None)


def test_llm_backing_raises_session2_stub():
    """Serves (C) firewall: even with a provider, the call path is Session 2."""
    backing = LLMBacking(prompt_template="{paper}")
    with pytest.raises(NotImplementedError, match="LLM call path is a Session 2"):
        backing.forward(torch.zeros(1), slot_values={"_llm_provider": object()})


def test_symbolic_backing_dispatches_rule():
    """Serves (B): symbolic backing dispatches to its rule."""

    def rule(context, slot_values):
        return context * 2, torch.ones_like(context)

    backing = SymbolicBacking(rule)
    output, alpha = backing.forward(torch.tensor([1.0, 2.0]))
    assert torch.allclose(output, torch.tensor([2.0, 4.0]))
    assert torch.all(alpha > 0)


def test_composite_backing_sequential_threads_outputs():
    """Serves combinatorial novelty: composites thread outputs sequentially."""

    def add_one(context, slot_values):
        return context + 1, torch.ones_like(context)

    latent_dim = 3
    child_a = Schema("a", SymbolicBacking(add_one), latent_dim)
    child_b = Schema("b", SymbolicBacking(add_one), latent_dim)
    backing = CompositeBacking([child_a, child_b], mode="sequential")
    output, alpha = backing.forward(torch.zeros(latent_dim))
    assert torch.allclose(output, torch.full((latent_dim,), 2.0))
    assert torch.all(alpha > 0)


def test_composite_backing_parallel_averages():
    """Serves combinatorial novelty: parallel composites average children."""

    def ret_two(context, slot_values):
        return torch.full_like(context, 2.0), torch.full_like(context, 2.0)

    def ret_four(context, slot_values):
        return torch.full_like(context, 4.0), torch.full_like(context, 4.0)

    latent_dim = 2
    child_a = Schema("a", SymbolicBacking(ret_two), latent_dim)
    child_b = Schema("b", SymbolicBacking(ret_four), latent_dim)
    backing = CompositeBacking([child_a, child_b], mode="parallel")
    output, alpha = backing.forward(torch.zeros(latent_dim))
    assert torch.allclose(output, torch.full((latent_dim,), 3.0))
    assert torch.allclose(alpha, torch.full((latent_dim,), 3.0))


def test_composite_backing_alpha_stays_positive():
    """Serves (B): parallel composite clamps alpha to preserve positivity."""

    def zero_alpha(context, slot_values):
        return context, torch.zeros_like(context)

    latent_dim = 2
    child_a = Schema("a", SymbolicBacking(zero_alpha), latent_dim)
    child_b = Schema("b", SymbolicBacking(zero_alpha), latent_dim)
    backing = CompositeBacking([child_a, child_b], mode="parallel")
    _, alpha = backing.forward(torch.zeros(latent_dim))
    assert torch.all(alpha > 0)


def test_composite_backing_rejects_unknown_mode():
    """Serves (D) falsifiability: invalid composition modes fail loudly."""
    with pytest.raises(ValueError, match="Unknown composite mode"):
        CompositeBacking([], mode="diagonal")


def test_schema_predict_returns_prediction_and_alpha():
    """Serves (B): Schema.predict returns (output, positive alpha)."""
    latent_dim = 4
    schema = _neural_schema(latent_dim=latent_dim)
    output, alpha = schema.predict(torch.randn(1, latent_dim))
    assert output.shape == (1, latent_dim)
    assert torch.all(alpha > 0)


def test_schema_ablation_returns_zeros():
    """Serves Phase 5 criterion 2: ablated schema returns zeros + flat alpha."""
    latent_dim = 4
    schema = _neural_schema(latent_dim=latent_dim)
    schema.ablate()
    output, alpha = schema.predict(torch.randn(3, latent_dim))
    assert output.shape == (3, latent_dim)
    assert torch.all(output == 0)
    assert torch.all(alpha == 1)


def test_schema_restore_reverses_ablation():
    """Serves Phase 5 criterion 2: ablation is reversible."""
    schema = _neural_schema()
    schema.ablate()
    assert schema.is_ablated
    schema.restore()
    assert not schema.is_ablated


def test_schema_activation_count_increments():
    """Serves (A): activation count tracks engagement for lifecycle/prune."""
    latent_dim = 4
    schema = _neural_schema(latent_dim=latent_dim)
    schema.predict(torch.randn(1, latent_dim))
    schema.predict(torch.randn(1, latent_dim))
    assert schema.meta.activation_count == 2


def test_schema_ablated_does_not_increment_activation():
    """Serves Phase 5 criterion 2: ablation removes a schema from the count."""
    latent_dim = 4
    schema = _neural_schema(latent_dim=latent_dim)
    schema.ablate()
    schema.predict(torch.randn(1, latent_dim))
    assert schema.meta.activation_count == 0


def test_schema_activate_in_uses_context_signature():
    """Serves (A): context-sensitive activation drives personalization."""
    latent_dim = 4
    signature = torch.tensor([1.0, 0.0, 0.0, 0.0])
    schema = _neural_schema(latent_dim=latent_dim, signature=signature)
    aligned = torch.tensor([2.0, 0.0, 0.0, 0.0])
    orthogonal = torch.tensor([0.0, 1.0, 0.0, 0.0])
    assert schema.activate_in(aligned) == pytest.approx(1.0, abs=1e-5)
    assert schema.activate_in(orthogonal) == pytest.approx(0.0, abs=1e-5)
    zero_sig_schema = _neural_schema(latent_dim=latent_dim)
    assert zero_sig_schema.activate_in(aligned) == 0.0


def test_schema_map_to_matches_slots_by_name():
    """Serves combinatorial novelty: naive Session 1 structure mapping."""
    latent_dim = 4
    slots_a = {"x": TypedSlot("x", "float"), "y": TypedSlot("y", "float")}
    slots_b = {"x": TypedSlot("x", "float"), "z": TypedSlot("z", "float")}
    schema_a = Schema(
        "a", NeuralBacking(nn.Linear(latent_dim, latent_dim), latent_dim), latent_dim, slots=slots_a
    )
    schema_b = Schema(
        "b", NeuralBacking(nn.Linear(latent_dim, latent_dim), latent_dim), latent_dim, slots=slots_b
    )
    mapping = schema_a.map_to(schema_b)
    assert mapping.slot_correspondences == [("x", "x")]
    assert mapping.confidence == pytest.approx(0.5)


def test_schema_specialize_records_parent_and_provenance():
    """Serves (C): specialization records parent and provenance (Phase 2)."""
    schema = _neural_schema()
    child = schema.specialize({"domain": "world-models"})
    assert child.meta.parents == [schema]
    assert child.meta.provenance.event == "specialize"
    assert "world-models" in child.meta.provenance.trigger_description
    assert child in schema.meta.children


def test_schema_generalize_returns_parent_when_available():
    """Serves (C): generalize lifts to an existing parent."""
    schema = _neural_schema()
    child = schema.specialize({"k": "v"})
    assert child.generalize() is schema


def test_schema_generalize_raises_without_parent():
    """Serves (D): standalone generalization is honestly deferred to Phase 2."""
    schema = _neural_schema()
    with pytest.raises(NotImplementedError):
        schema.generalize()


def test_schema_compose_creates_composite():
    """Serves combinatorial novelty: compose builds a composite schema."""
    latent_dim = 3
    schema_a = _neural_schema("a", latent_dim)
    schema_b = _neural_schema("b", latent_dim)
    composite = schema_a.compose(schema_b, mode="sequential")
    assert isinstance(composite.backing, CompositeBacking)
    assert schema_a in composite.meta.parents
    assert schema_b in composite.meta.parents
    assert composite.meta.provenance.event == "compose"


def test_schema_confidence_trajectory_grows():
    """Serves calibration track: predict appends to confidence_trajectory."""
    latent_dim = 4
    schema = _neural_schema(latent_dim=latent_dim)
    assert schema.meta.confidence_trajectory == []
    schema.predict(torch.randn(1, latent_dim))
    assert len(schema.meta.confidence_trajectory) == 1


def test_schema_meta_has_unique_ids():
    """Serves (D): every schema has an inspectable unique id."""
    assert _neural_schema().meta.schema_id != _neural_schema().meta.schema_id


def test_schema_repr_includes_referent_and_backing_type():
    """Serves (D): schemas are inspectable at a glance."""
    representation = repr(_neural_schema("novelty-vs-KB"))
    assert "novelty-vs-KB" in representation
    assert "neural" in representation


def test_schema_parameters_include_backing_module():
    """Serves (B): a neural schema exposes its backing's trainable params."""
    latent_dim = 4
    schema = _neural_schema(latent_dim=latent_dim)
    param_shapes = {tuple(p.shape) for p in schema.parameters()}
    assert (latent_dim, latent_dim) in param_shapes  # wrapped Linear weight
    assert (latent_dim,) in param_shapes  # the schema embedding


def test_schema_compose_combines_slots():
    """Serves combinatorial novelty: composition unions the parents' slots."""
    latent_dim = 3
    schema_a = Schema(
        "a",
        NeuralBacking(nn.Linear(latent_dim, latent_dim), latent_dim),
        latent_dim,
        slots={"x": TypedSlot("x", "float")},
    )
    schema_b = Schema(
        "b",
        NeuralBacking(nn.Linear(latent_dim, latent_dim), latent_dim),
        latent_dim,
        slots={"y": TypedSlot("y", "float")},
    )
    composite = schema_a.compose(schema_b)
    assert set(composite.slots) == {"x", "y"}
