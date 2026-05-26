"""Structured-output protocol — the (C) firewall at the LLM-substrate boundary.

Strange-loop role
-----------------
This is where the (C) architectural firewall is enforced at the boundary
between the orchestrator and the frozen LLM substrate. The framework dictates
the output shape; the LLM must conform. If a response cannot be parsed and
validated against the protocol, we **raise** — we never soft-coerce, pad
missing fields, or hallucinate alpha values. An LLM that cannot produce
conforming output is not used; it does not get to renegotiate the protocol.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor


class StructuredOutputParseError(Exception):
    """Raised when an LLM response cannot be parsed/validated against the protocol."""


@dataclass
class StructuredOutputProtocol:
    """Specifies the JSON output contract an LLM response must satisfy.

    Parameters
    ----------
    latent_dim:
        Required length of the ``output`` and ``alpha`` lists.
    """

    latent_dim: int

    @property
    def instruction_text(self) -> str:
        """Deterministic JSON-output instructions appended to LLM prompts."""
        return (
            "Respond with a single JSON object and nothing else. The object must "
            "have these keys:\n"
            f'- "output": a list of exactly {self.latent_dim} finite floats '
            "(the latent semantic output).\n"
            f'- "alpha": a list of exactly {self.latent_dim} strictly positive '
            "finite floats (Dirichlet concentration parameters; every entry > 0).\n"
            '- "reasoning": an optional string explaining the assessment.\n'
            '- "confidence_summary": an optional short string summarizing confidence.\n'
            "Return only the JSON object."
        )


def _extract_json_object(text: str) -> dict:
    """Liberally extract a JSON object from ``text``.

    Tries the whole text first; then the first ``{`` to its matching ``}``.
    Raises ``StructuredOutputParseError`` if neither yields a JSON object.
    """
    try:
        whole = json.loads(text)
        if isinstance(whole, dict):
            return whole
    except (json.JSONDecodeError, TypeError):
        pass

    start = text.find("{")
    if start == -1:
        raise StructuredOutputParseError(
            f"No JSON object found in response: {text[:200]!r}"
        )
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                snippet = text[start : index + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError as exc:
                    raise StructuredOutputParseError(
                        f"Brace-delimited span was not valid JSON: {snippet[:200]!r}"
                    ) from exc
    raise StructuredOutputParseError(
        f"Unbalanced braces; could not extract a JSON object: {text[:200]!r}"
    )


def parse_structured_response(
    text: str, protocol: StructuredOutputProtocol
) -> tuple[Tensor, Tensor, dict]:
    """Parse and validate an LLM response into ``(output, alpha, metadata)``.

    Validates that ``output`` and ``alpha`` are lists of length
    ``protocol.latent_dim``, all entries are finite, and every ``alpha`` entry
    is strictly positive. Raises ``StructuredOutputParseError`` otherwise.
    """
    obj = _extract_json_object(text)

    if "output" not in obj or "alpha" not in obj:
        raise StructuredOutputParseError(
            f"Response JSON missing required 'output'/'alpha' keys: {text[:200]!r}"
        )
    output = obj["output"]
    alpha = obj["alpha"]

    if not isinstance(output, list) or len(output) != protocol.latent_dim:
        raise StructuredOutputParseError(
            f"'output' must be a list of length {protocol.latent_dim}; got {output!r}"
        )
    if not isinstance(alpha, list) or len(alpha) != protocol.latent_dim:
        raise StructuredOutputParseError(
            f"'alpha' must be a list of length {protocol.latent_dim}; got {alpha!r}"
        )

    try:
        output_vals = [float(x) for x in output]
        alpha_vals = [float(x) for x in alpha]
    except (TypeError, ValueError) as exc:
        raise StructuredOutputParseError(
            f"'output'/'alpha' entries must be numeric: {text[:200]!r}"
        ) from exc

    if not all(isfinite(x) for x in output_vals):
        raise StructuredOutputParseError(f"'output' has non-finite values: {output_vals!r}")
    if not all(isfinite(x) for x in alpha_vals):
        raise StructuredOutputParseError(f"'alpha' has non-finite values: {alpha_vals!r}")
    if not all(x > 0 for x in alpha_vals):
        raise StructuredOutputParseError(
            f"'alpha' entries must all be strictly positive; got {alpha_vals!r}"
        )

    metadata = {
        "reasoning": obj.get("reasoning"),
        "confidence_summary": obj.get("confidence_summary"),
        "raw_json": obj,
    }
    return (
        torch.tensor(output_vals, dtype=torch.float32),
        torch.tensor(alpha_vals, dtype=torch.float32),
        metadata,
    )
