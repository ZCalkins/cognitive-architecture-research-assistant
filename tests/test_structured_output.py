"""Tests for the structured-output protocol.

Each test serves the (C) firewall: the framework defines the output protocol
and the LLM conforms; non-conformance is a hard error, never silent coercion.
"""

import pytest
import torch

from src.schemas.structured_output import (
    StructuredOutputParseError,
    StructuredOutputProtocol,
    parse_structured_response,
)


def test_protocol_instruction_text_includes_latent_dim():
    """Serves (C) firewall: the protocol announces the exact required shape."""
    protocol = StructuredOutputProtocol(latent_dim=4)
    text = protocol.instruction_text
    assert "4" in text
    assert "output" in text
    assert "alpha" in text


def test_parse_valid_response():
    """Serves (C) firewall: well-formed JSON parses to (output, alpha, meta)."""
    protocol = StructuredOutputProtocol(latent_dim=2)
    output, alpha, metadata = parse_structured_response(
        '{"output": [0.1, 0.2], "alpha": [1.0, 2.0]}', protocol
    )
    assert torch.allclose(output, torch.tensor([0.1, 0.2]))
    assert torch.allclose(alpha, torch.tensor([1.0, 2.0]))
    assert "raw_json" in metadata


def test_parse_response_with_surrounding_text():
    """Serves (C) firewall: JSON embedded in prose is still extracted."""
    protocol = StructuredOutputProtocol(latent_dim=1)
    text = 'Sure! Here is my answer: {"output": [0.5], "alpha": [1.5]} Hope it helps.'
    output, alpha, _ = parse_structured_response(text, protocol)
    assert torch.allclose(output, torch.tensor([0.5]))
    assert torch.allclose(alpha, torch.tensor([1.5]))


def test_parse_rejects_wrong_output_length():
    """Serves (C) firewall: wrong-length output is a hard error."""
    protocol = StructuredOutputProtocol(latent_dim=3)
    with pytest.raises(StructuredOutputParseError):
        parse_structured_response('{"output": [0.1, 0.2], "alpha": [1, 1, 1]}', protocol)


def test_parse_rejects_wrong_alpha_length():
    """Serves (C) firewall: wrong-length alpha is a hard error."""
    protocol = StructuredOutputProtocol(latent_dim=3)
    with pytest.raises(StructuredOutputParseError):
        parse_structured_response('{"output": [0.1, 0.2, 0.3], "alpha": [1, 1]}', protocol)


def test_parse_rejects_non_positive_alpha():
    """Serves (C) firewall: alpha must be strictly positive (EDL invariant)."""
    protocol = StructuredOutputProtocol(latent_dim=2)
    with pytest.raises(StructuredOutputParseError):
        parse_structured_response('{"output": [0.1, 0.2], "alpha": [1.0, 0.0]}', protocol)


def test_parse_rejects_malformed_json():
    """Serves (C) firewall: syntactically invalid JSON is a hard error."""
    protocol = StructuredOutputProtocol(latent_dim=1)
    with pytest.raises(StructuredOutputParseError):
        parse_structured_response("this is not json at all", protocol)


def test_parse_extracts_optional_metadata():
    """Serves (C) firewall: optional reasoning/confidence_summary surface in meta."""
    protocol = StructuredOutputProtocol(latent_dim=1)
    text = (
        '{"output": [0.1], "alpha": [1.0], "reasoning": "novel substrate", '
        '"confidence_summary": "high"}'
    )
    _, _, metadata = parse_structured_response(text, protocol)
    assert metadata["reasoning"] == "novel substrate"
    assert metadata["confidence_summary"] == "high"
