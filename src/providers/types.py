"""Request/response value types for the ``LLMProvider`` abstraction.

These types are vendor-neutral. Each concrete provider translates between
these and its SDK's native request/response shapes, so the orchestrator never
depends on a vendor SDK directly (the F1 framework choice in ARCHITECTURE.md).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMRequest:
    """A vendor-neutral completion request.

    Parameters
    ----------
    messages:
        List of ``{"role": ..., "content": ...}`` message dicts.
    model:
        Model identifier (vendor-specific string).
    temperature:
        Sampling temperature.
    max_tokens:
        Maximum tokens to generate.
    system:
        Optional system prompt. Providers map this to the vendor's native
        mechanism (top-level field for Anthropic, prepended message for
        Ollama).
    metadata:
        Optional opaque metadata passed through by the caller.
    """

    messages: list[dict]
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    system: str | None = None
    metadata: dict | None = None


@dataclass
class LLMResponse:
    """A vendor-neutral completion response.

    Parameters
    ----------
    text:
        The generated text.
    model:
        The model that produced the response.
    input_tokens:
        Prompt token count.
    output_tokens:
        Generated token count.
    stop_reason:
        Why generation stopped (vendor-specific string).
    raw:
        The full vendor-specific response, as an opaque dict.
    """

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str
    raw: dict
