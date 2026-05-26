"""Anthropic (Claude) provider — the primary LLM substrate.

Clients are instantiated lazily on first call so that importing this module
and constructing the provider require neither network connectivity nor API
credentials. Network/credential failures surface as the Anthropic SDK's own
exceptions; they are not caught here.
"""

from __future__ import annotations

import os
from typing import Any

import anthropic

from src.providers.base import LLMProvider
from src.providers.types import LLMRequest, LLMResponse


class AnthropicProvider(LLMProvider):
    """Wraps the Anthropic SDK.

    Parameters
    ----------
    api_key:
        API key. Defaults to the ``ANTHROPIC_API_KEY`` environment variable.
        Not required to construct the provider; only to make a real call.
    """

    provider_name = "anthropic"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client: anthropic.Anthropic | None = None
        self._async_client: anthropic.AsyncAnthropic | None = None

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def _get_async_client(self) -> anthropic.AsyncAnthropic:
        if self._async_client is None:
            self._async_client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._async_client

    def _build_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if request.system is not None:
            kwargs["system"] = request.system
        return kwargs

    @staticmethod
    def _translate_response(response: Any) -> LLMResponse:
        text = "".join(getattr(block, "text", "") for block in response.content)
        return LLMResponse(
            text=text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
            raw=response.model_dump(),
        )

    def complete(self, request: LLMRequest) -> LLMResponse:
        client = self._get_client()
        response = client.messages.create(**self._build_kwargs(request))
        return self._translate_response(response)

    async def complete_async(self, request: LLMRequest) -> LLMResponse:
        client = self._get_async_client()
        response = await client.messages.create(**self._build_kwargs(request))
        return self._translate_response(response)
