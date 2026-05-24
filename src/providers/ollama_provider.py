"""Ollama provider — the secondary LLM substrate (local models).

Used for the Phase 5 criterion 5 backend-swap test. Clients are instantiated
lazily on first call so that importing this module and constructing the
provider do not require a reachable Ollama host. Connection failures surface
as the Ollama SDK's own exceptions.

Ollama has no top-level system parameter, so a request's ``system`` prompt is
prepended to the message list as a ``system``-role message.
"""

from __future__ import annotations

from typing import Any

import ollama

from src.providers.base import LLMProvider
from src.providers.types import LLMRequest, LLMResponse


class OllamaProvider(LLMProvider):
    """Wraps the Ollama SDK.

    Parameters
    ----------
    host:
        Ollama host URL. Defaults to ``http://localhost:11434``.
    """

    provider_name = "ollama"

    def __init__(self, host: str | None = None) -> None:
        self._host = host or "http://localhost:11434"
        self._client: ollama.Client | None = None
        self._async_client: ollama.AsyncClient | None = None

    def _get_client(self) -> ollama.Client:
        if self._client is None:
            self._client = ollama.Client(host=self._host)
        return self._client

    def _get_async_client(self) -> ollama.AsyncClient:
        if self._async_client is None:
            self._async_client = ollama.AsyncClient(host=self._host)
        return self._async_client

    @staticmethod
    def _build_messages(request: LLMRequest) -> list[dict]:
        messages = list(request.messages)
        if request.system is not None:
            messages = [{"role": "system", "content": request.system}, *messages]
        return messages

    @staticmethod
    def _options(request: LLMRequest) -> dict[str, Any]:
        return {"temperature": request.temperature, "num_predict": request.max_tokens}

    @staticmethod
    def _translate_response(response: Any) -> LLMResponse:
        return LLMResponse(
            text=response["message"]["content"],
            model=response.get("model", ""),
            input_tokens=response.get("prompt_eval_count", 0),
            output_tokens=response.get("eval_count", 0),
            stop_reason=response.get("done_reason", "unknown"),
            raw=dict(response),
        )

    def complete(self, request: LLMRequest) -> LLMResponse:
        client = self._get_client()
        response = client.chat(
            model=request.model,
            messages=self._build_messages(request),
            options=self._options(request),
        )
        return self._translate_response(response)

    async def complete_async(self, request: LLMRequest) -> LLMResponse:
        client = self._get_async_client()
        response = await client.chat(
            model=request.model,
            messages=self._build_messages(request),
            options=self._options(request),
        )
        return self._translate_response(response)
