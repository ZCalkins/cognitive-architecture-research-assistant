"""Tests for the real LLMBacking-backed seed specialists (Session 4).

LLMs are object-level substrate (CLAUDE.md principle 3). These pin the
specialist contract — model assignment, slots, latent_dim=1 score/confidence —
and the (C) firewall (the framework's protocol parses the LLM's response;
non-conformance is a hard error). Hermetic: the provider is faked, no Ollama.
"""

import pytest
import torch

from src.providers import LLMResponse
from src.schemas import LLMBacking
from src.specialists import (
    SPECIALIST_NAMES,
    build_specialist,
    build_v0_specialist_registry,
)

_CONFIG = {
    "embedding": {"dimension": 1024},
    "specialist_models": {"triage_light": "llama-light", "triage_heavy": "mistral-heavy"},
    "specialist_assignments": {
        "novelty_vs_kb": "triage_light",
        "relevance_to_projects": "triage_light",
        "citation_graph_position": "triage_light",
        "author_history": "triage_light",
        "methodological_rigor": "triage_heavy",
        "theoretical_claim_evaluator": "triage_heavy",
    },
}


class _FakeProvider:
    def __init__(self, text='{"output": [0.7], "alpha": [3.0]}'):
        self.text = text
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return LLMResponse(
            text=self.text,
            model=request.model,
            input_tokens=5,
            output_tokens=5,
            stop_reason="end_turn",
            raw={},
        )


def _slot_values(provider):
    return {
        "_llm_provider": provider,
        "title": "A Title",
        "abstract": "An abstract.",
        "authors": "Jane Doe; John Roe",
        "categories": "cs.AI",
        "kb_neighbors": "- Similar paper",
        "projects": "strange-loop",
    }


def test_registry_builds_six_specialists():
    """Serves (C): the seed registry has the six v0 specialists."""
    registry = build_v0_specialist_registry(_CONFIG)
    assert len(registry) == 6
    assert {schema.meta.referent for schema in registry.all()} == set(SPECIALIST_NAMES)


def test_specialists_are_llm_backed_latent_dim_one():
    """Serves (C) firewall: each specialist is LLMBacking with a 1-d score output."""
    for name in SPECIALIST_NAMES:
        schema = build_specialist(name, _CONFIG)
        assert isinstance(schema.backing, LLMBacking)
        assert schema.latent_dim == 1
        assert schema.backing.latent_dim == 1


def test_specialist_context_signature_matches_embedding_dim():
    """Serves (B): routing signature is sized to the query embedding, not output."""
    schema = build_specialist("novelty_vs_kb", _CONFIG)
    assert schema.context_signature.shape == (1024,)


def test_model_assignment_light_and_heavy():
    """Serves (C): light/heavy specialists get the configured models."""
    assert build_specialist("novelty_vs_kb", _CONFIG).backing.model == "llama-light"
    assert build_specialist("methodological_rigor", _CONFIG).backing.model == "mistral-heavy"


def test_specialist_predict_parses_provider_response():
    """Serves (C) firewall: the framework parses the LLM response into (output, alpha)."""
    provider = _FakeProvider()
    schema = build_specialist("novelty_vs_kb", _CONFIG)
    output, alpha = schema.predict(torch.zeros(1024), _slot_values(provider))
    assert torch.allclose(output, torch.tensor([0.7]))
    assert torch.allclose(alpha, torch.tensor([3.0]))
    assert len(provider.requests) == 1


def test_specialist_prompt_includes_slots_and_protocol():
    """Serves (C) firewall: the rendered prompt carries the task slots + protocol."""
    provider = _FakeProvider()
    schema = build_specialist("novelty_vs_kb", _CONFIG)
    schema.predict(torch.zeros(1024), _slot_values(provider))
    content = provider.requests[0].messages[0]["content"]
    assert "A Title" in content
    assert "Similar paper" in content
    assert "output" in content
    assert "alpha" in content


def test_data_starved_specialists_declare_available_slots():
    """Serves (D): citation/author specialists declare only available-metadata slots."""
    assert set(build_specialist("citation_graph_position", _CONFIG).slots) == {
        "title",
        "authors",
        "categories",
    }
    assert set(build_specialist("author_history", _CONFIG).slots) == {"authors"}


def test_specialist_missing_slot_raises():
    """Serves (C) firewall: a missing required slot is a hard error, not a guess."""
    schema = build_specialist("relevance_to_projects", _CONFIG)
    slots = {"_llm_provider": _FakeProvider(), "title": "T", "abstract": "A"}
    with pytest.raises(KeyError):
        schema.predict(torch.zeros(1024), slots)
