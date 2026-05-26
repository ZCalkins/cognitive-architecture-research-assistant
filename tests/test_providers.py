"""Tests for the LLMProvider abstraction.

LLMs are object-level substrate (CLAUDE.md principle 3). A thin, uniform
provider seam is what makes the Phase 5 criterion 5 backend-swap test a
one-line change, so these tests pin the vendor-translation behavior.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.providers import (
    AnthropicProvider,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    OllamaProvider,
)


def test_llm_request_response_dataclasses():
    """Serves (D): vendor-neutral request/response value types construct."""
    request = LLMRequest(messages=[{"role": "user", "content": "hi"}], model="m")
    assert request.temperature == 0.7
    assert request.max_tokens == 4096
    response = LLMResponse(
        text="t", model="m", input_tokens=1, output_tokens=2, stop_reason="end_turn", raw={}
    )
    assert response.text == "t"


def test_anthropic_provider_importable_without_credentials(monkeypatch):
    """Serves Phase 5 criterion 5: provider constructs without credentials."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider = AnthropicProvider()
    assert isinstance(provider, LLMProvider)
    assert provider.provider_name == "anthropic"


def test_ollama_provider_importable_without_host():
    """Serves Phase 5 criterion 5: secondary provider constructs without host."""
    provider = OllamaProvider()
    assert isinstance(provider, LLMProvider)
    assert provider.provider_name == "ollama"


def test_anthropic_provider_translates_system_to_top_level():
    """Serves (C) substrate: system goes to the top-level field, not messages."""
    block = MagicMock()
    block.text = "hello"
    response = MagicMock()
    response.content = [block]
    response.model = "claude-sonnet-4-5"
    response.usage.input_tokens = 5
    response.usage.output_tokens = 7
    response.stop_reason = "end_turn"
    response.model_dump.return_value = {"ok": True}
    client = MagicMock()
    client.messages.create.return_value = response

    with patch("anthropic.Anthropic", return_value=client):
        provider = AnthropicProvider(api_key="test")
        request = LLMRequest(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-sonnet-4-5",
            system="be terse",
        )
        result = provider.complete(request)

    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["system"] == "be terse"
    assert all(message.get("role") != "system" for message in kwargs["messages"])
    assert result.text == "hello"
    assert result.input_tokens == 5
    assert result.output_tokens == 7


def test_ollama_provider_prepends_system_message():
    """Serves (C) substrate: Ollama has no system field, so it is prepended."""
    response = {
        "message": {"content": "hi"},
        "model": "llama3.1",
        "prompt_eval_count": 3,
        "eval_count": 4,
        "done_reason": "stop",
    }
    client = MagicMock()
    client.chat.return_value = response

    with patch("ollama.Client", return_value=client):
        provider = OllamaProvider()
        request = LLMRequest(
            messages=[{"role": "user", "content": "hi"}],
            model="llama3.1",
            system="be terse",
        )
        result = provider.complete(request)

    messages = client.chat.call_args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "be terse"}
    assert messages[1] == {"role": "user", "content": "hi"}
    assert result.text == "hi"
    assert result.input_tokens == 3
    assert result.output_tokens == 4
    assert result.stop_reason == "stop"


def test_anthropic_no_system_omits_system_kwarg():
    """Serves (C) substrate: with no system prompt, no system kwarg is sent."""
    block = MagicMock()
    block.text = "ok"
    response = MagicMock()
    response.content = [block]
    response.model = "claude-sonnet-4-5"
    response.usage.input_tokens = 1
    response.usage.output_tokens = 1
    response.stop_reason = "end_turn"
    response.model_dump.return_value = {}
    client = MagicMock()
    client.messages.create.return_value = response

    with patch("anthropic.Anthropic", return_value=client):
        provider = AnthropicProvider(api_key="test")
        provider.complete(
            LLMRequest(messages=[{"role": "user", "content": "hi"}], model="claude-sonnet-4-5")
        )

    assert "system" not in client.messages.create.call_args.kwargs


@pytest.mark.network
def test_anthropic_real_call_smoke():
    """Serves Phase 5 criterion 5: real Anthropic round-trip (gated)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    provider = AnthropicProvider()
    request = LLMRequest(
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        model="claude-sonnet-4-5",
        max_tokens=16,
    )
    result = provider.complete(request)
    assert isinstance(result.text, str)


@pytest.mark.network
def test_ollama_real_call_smoke():
    """Serves Phase 5 criterion 5: real Ollama round-trip (gated)."""
    provider = OllamaProvider()
    try:
        provider._get_client().list()
    except Exception:
        pytest.skip("Ollama not reachable")
    request = LLMRequest(
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        model="llama3.1",
        max_tokens=16,
    )
    result = provider.complete(request)
    assert isinstance(result.text, str)
