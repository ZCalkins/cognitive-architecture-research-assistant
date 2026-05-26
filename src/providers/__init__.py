"""LLMProvider abstraction — thin vendor-SDK wrappers (F1 framework choice).

LLMs are object-level substrate. See ARCHITECTURE.md "Multi-agent framework"
and CLAUDE.md principle 3.
"""

from src.providers.anthropic_provider import AnthropicProvider
from src.providers.base import LLMProvider
from src.providers.ollama_provider import OllamaProvider
from src.providers.types import LLMRequest, LLMResponse

__all__ = [
    "AnthropicProvider",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "OllamaProvider",
]
