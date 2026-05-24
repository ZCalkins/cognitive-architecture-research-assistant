"""The ``LLMProvider`` abstraction.

LLMs are object-level substrate (CLAUDE.md principle 3). The provider
interface is a thin, vendor-neutral wrapper so that swapping the underlying
LLM backend (Phase 5 criterion 5) is a one-line change and never leaks vendor
SDK types into the orchestrator.

Strange-loop role
-----------------
This interface is the seam at which the orchestrator delegates object-level
work to frozen LLMs without surrendering meta-cognition to them. The provider
never makes routing or self-modeling decisions; it only completes a request.
The backend-swap method (Phase 5 criterion 5) depends on this seam being thin
and uniform across vendors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.providers.types import LLMRequest, LLMResponse


class LLMProvider(ABC):
    """Vendor-neutral completion provider."""

    @property
    def provider_name(self) -> str:
        """Short provider identifier (subclasses override)."""
        raise NotImplementedError

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Complete ``request`` synchronously."""
        ...

    @abstractmethod
    async def complete_async(self, request: LLMRequest) -> LLMResponse:
        """Complete ``request`` asynchronously."""
        ...
