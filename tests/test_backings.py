"""Tests for the LLMBacking call path (Session 2).

LLMs are object-level substrate (CLAUDE.md principle 3). These tests pin the
(C) firewall at the substrate boundary: the framework builds the prompt,
appends the output protocol, and parses/validates the response — retrying once,
then failing hard rather than soft-coercing.
"""

import pytest
import torch

from src.instrumentation.probes import Probe
from src.providers import LLMResponse
from src.schemas import LLMBacking
from src.schemas.structured_output import StructuredOutputParseError


class _FakeProvider:
    """Returns queued response texts; records the requests it received."""

    def __init__(self, texts):
        self._texts = list(texts)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return LLMResponse(
            text=self._texts.pop(0),
            model=request.model,
            input_tokens=11,
            output_tokens=22,
            stop_reason="end_turn",
            raw={},
        )


def test_llm_backing_forward_calls_provider():
    """Serves (C) firewall: forward invokes the provider and parses the result."""
    backing = LLMBacking(prompt_template="Assess {paper}", latent_dim=2)
    provider = _FakeProvider(['{"output": [0.1, 0.2], "alpha": [1.0, 2.0]}'])
    output, alpha = backing.forward(
        torch.zeros(2), slot_values={"_llm_provider": provider, "paper": "X"}
    )
    assert len(provider.requests) == 1
    assert torch.allclose(output, torch.tensor([0.1, 0.2]))
    assert torch.allclose(alpha, torch.tensor([1.0, 2.0]))


def test_llm_backing_forward_renders_template_with_slot_values():
    """Serves (A): {slot} placeholders are filled from slot_values."""
    backing = LLMBacking(prompt_template="Paper titled {title}", latent_dim=1)
    provider = _FakeProvider(['{"output": [0.5], "alpha": [1.5]}'])
    backing.forward(
        torch.zeros(1), slot_values={"_llm_provider": provider, "title": "Mamba"}
    )
    assert "Mamba" in provider.requests[0].messages[0]["content"]


def test_llm_backing_forward_appends_protocol_instructions():
    """Serves (C) firewall: the output protocol is appended to the prompt."""
    backing = LLMBacking(prompt_template="x", latent_dim=1)
    provider = _FakeProvider(['{"output": [0.5], "alpha": [1.5]}'])
    backing.forward(torch.zeros(1), slot_values={"_llm_provider": provider})
    content = provider.requests[0].messages[0]["content"]
    assert "JSON object" in content
    assert "alpha" in content


def test_llm_backing_forward_retries_once_on_parse_failure():
    """Serves (C) firewall: one corrective retry on a parse failure."""
    backing = LLMBacking(prompt_template="x", latent_dim=1)
    provider = _FakeProvider(["garbage, no json", '{"output": [0.5], "alpha": [1.5]}'])
    output, _ = backing.forward(torch.zeros(1), slot_values={"_llm_provider": provider})
    assert len(provider.requests) == 2
    assert torch.allclose(output, torch.tensor([0.5]))


def test_llm_backing_forward_raises_after_two_failed_attempts():
    """Serves (C) firewall: two non-conforming responses fail hard."""
    backing = LLMBacking(prompt_template="x", latent_dim=1)
    provider = _FakeProvider(["nope", "still nope"])
    with pytest.raises(StructuredOutputParseError):
        backing.forward(torch.zeros(1), slot_values={"_llm_provider": provider})
    assert len(provider.requests) == 2


def test_llm_backing_forward_excludes_framework_slots_from_rendering():
    """Serves (C) firewall: `_`-prefixed framework slots are not interpolated."""
    backing = LLMBacking(prompt_template="topic={topic}", latent_dim=1)
    provider = _FakeProvider(['{"output": [0.1], "alpha": [1.0]}'])
    backing.forward(
        torch.zeros(1),
        slot_values={"_llm_provider": provider, "topic": "AI", "_system_prompt": "secret"},
    )
    content = provider.requests[0].messages[0]["content"]
    assert "topic=AI" in content
    assert "secret" not in content


def test_llm_backing_forward_records_to_probe_when_provided():
    """Serves Phase 5 criterion (D): token counts and response length are probed."""
    backing = LLMBacking(prompt_template="x", latent_dim=1)
    provider = _FakeProvider(['{"output": [0.1], "alpha": [1.0]}'])
    probe = Probe()
    backing.forward(
        torch.zeros(1), slot_values={"_llm_provider": provider, "_probe": probe}
    )
    assert probe.get_scalars("llm_input_tokens") == [11]
    assert probe.get_scalars("llm_output_tokens") == [22]
    assert len(probe.get("llm_response_length")) == 1


def test_llm_backing_forward_passes_system_prompt():
    """Serves (C) substrate: _system_prompt becomes request.system."""
    backing = LLMBacking(prompt_template="x", latent_dim=1)
    provider = _FakeProvider(['{"output": [0.1], "alpha": [1.0]}'])
    backing.forward(
        torch.zeros(1),
        slot_values={"_llm_provider": provider, "_system_prompt": "Be precise."},
    )
    assert provider.requests[0].system == "Be precise."


def test_llm_backing_missing_provider_raises_runtime_error():
    """Serves (C) firewall: a missing provider is a config error, not a stub."""
    backing = LLMBacking(prompt_template="x", latent_dim=1)
    with pytest.raises(RuntimeError, match="requires _llm_provider"):
        backing.forward(torch.zeros(1), slot_values=None)
