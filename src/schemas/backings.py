"""Polymorphic backing layer — heterogeneous object-level computation.

A schema's ``backing`` is its implementation. Four backing types are
supported: learned neural modules, frozen LLMs invoked by prompt, symbolic
rules/programs, and composites over other schemas.

Every backing's ``forward`` returns ``(output, dirichlet_alpha)``. The
Dirichlet alpha is the load-bearing invariant for the EDL / Subjective Logic
machinery introduced in Phase 3: **all alpha entries must be strictly
positive**. Each backing is responsible for preserving that.

Strange-loop role
-----------------
Polymorphic backings let the same self-schema + hypernetwork machinery
(Phase 3) operate uniformly over heterogeneous object-level computation. The
hypernetwork emits *typed* modulations that branch on ``backing_type`` —
weight deltas for ``NeuralBacking``, prompt-prefix / temperature / model
selection for ``LLMBacking``, rule-weighting for ``SymbolicBacking``, subgraph
routing weights for ``CompositeBacking``. Because the orchestrator never needs
to special-case a backing to *route* to it, the strange loop closes over a
single uniform schema interface regardless of what each schema is made of.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from src.schemas.structured_output import (
    StructuredOutputParseError,
    StructuredOutputProtocol,
    parse_structured_response,
)

if TYPE_CHECKING:
    from src.schemas.base import Schema


class SchemaBacking(ABC):
    """Abstract polymorphic implementation of a schema.

    Subclasses override ``forward`` to return ``(output, dirichlet_alpha)``
    where every entry of ``dirichlet_alpha`` is strictly positive.
    """

    backing_type: str = "abstract"

    @abstractmethod
    def forward(
        self, context: Tensor, slot_values: dict[str, Any] | None = None
    ) -> tuple[Tensor, Tensor]:
        """Compute ``(output, dirichlet_alpha)`` for ``context``."""
        ...


class NeuralBacking(SchemaBacking):
    """A learned-from-scratch neural module.

    The Dirichlet alpha is derived as ``ones_like(output) + output.abs()``,
    which is strictly positive by construction (every entry >= 1).
    """

    backing_type = "neural"

    def __init__(self, module: nn.Module, latent_dim: int) -> None:
        self.module = module
        self.latent_dim = latent_dim

    def forward(
        self, context: Tensor, slot_values: dict[str, Any] | None = None
    ) -> tuple[Tensor, Tensor]:
        output = self.module(context)
        alpha = torch.ones_like(output) + output.abs()
        return output, alpha

    def parameters(self) -> Iterator[nn.Parameter]:
        """Delegate to the wrapped module's parameters."""
        return self.module.parameters()


class LLMBacking(SchemaBacking):
    """A frozen LLM invoked by prompt template.

    The call path (Session 2): the provider (passed via
    ``slot_values["_llm_provider"]``) is asked to complete a prompt built from
    ``prompt_template`` plus the structured-output protocol instructions; the
    response is parsed and validated into ``(output, dirichlet_alpha)``. The
    template may use ``{slot_name}`` placeholders for any slot value;
    framework slots (prefixed ``_``, e.g. ``_llm_provider``, ``_system_prompt``,
    ``_probe``) are reserved and never interpolated into the prompt.
    """

    backing_type = "llm"

    def __init__(
        self,
        prompt_template: str,
        model: str = "claude-sonnet-4-5",
        default_temperature: float = 0.7,
        latent_dim: int = 0,
    ) -> None:
        self.prompt_template = prompt_template
        self.model = model
        self.default_temperature = default_temperature
        self.latent_dim = latent_dim

    def forward(
        self, context: Tensor, slot_values: dict[str, Any] | None = None
    ) -> tuple[Tensor, Tensor]:
        from src.providers.types import LLMRequest

        slot_values = slot_values or {}
        provider = slot_values.get("_llm_provider")
        if provider is None:
            raise RuntimeError(
                "LLMBacking requires _llm_provider in slot_values. "
                "Wire through the orchestrator."
            )

        protocol = StructuredOutputProtocol(latent_dim=self.latent_dim)
        render_values = {k: v for k, v in slot_values.items() if not k.startswith("_")}
        try:
            rendered = self.prompt_template.format(**render_values)
        except KeyError as exc:
            raise KeyError(
                f"prompt_template references missing slot {exc}; "
                f"available slots: {sorted(render_values)}"
            ) from exc

        system_prompt = slot_values.get("_system_prompt")
        probe = slot_values.get("_probe")

        def _call(message: str) -> str:
            request = LLMRequest(
                messages=[{"role": "user", "content": message}],
                model=self.model,
                temperature=self.default_temperature,
                system=system_prompt,
            )
            response = provider.complete(request)
            if probe is not None:
                probe.record("llm_response_length", torch.tensor([len(response.text)]))
                probe.record_scalar("llm_input_tokens", response.input_tokens)
                probe.record_scalar("llm_output_tokens", response.output_tokens)
            return response.text

        first_text = _call(rendered + "\n\n" + protocol.instruction_text)
        try:
            output_tensor, alpha_tensor, _ = parse_structured_response(first_text, protocol)
            return output_tensor, alpha_tensor
        except StructuredOutputParseError:
            corrective = (
                "Your previous response could not be parsed as the required JSON "
                f"schema. Respond again, strictly conforming to:\n{protocol.instruction_text}"
            )
            second_text = _call(corrective)
            try:
                output_tensor, alpha_tensor, _ = parse_structured_response(
                    second_text, protocol
                )
                return output_tensor, alpha_tensor
            except StructuredOutputParseError as exc:
                raise StructuredOutputParseError(
                    "LLMBacking failed to parse a conforming response after two "
                    f"attempts.\nFirst attempt: {first_text[:200]!r}\n"
                    f"Second attempt: {second_text[:200]!r}"
                ) from exc


class SymbolicBacking(SchemaBacking):
    """A symbolic rule or program.

    The rule is any callable mapping ``(context, slot_values)`` to
    ``(output, dirichlet_alpha)``; preserving alpha positivity is the rule's
    responsibility.
    """

    backing_type = "symbolic"

    def __init__(
        self,
        rule: Callable[[Tensor, dict[str, Any] | None], tuple[Tensor, Tensor]],
    ) -> None:
        self.rule = rule

    def forward(
        self, context: Tensor, slot_values: dict[str, Any] | None = None
    ) -> tuple[Tensor, Tensor]:
        return self.rule(context, slot_values)


class CompositeBacking(SchemaBacking):
    """A composite over other schemas.

    Two composition modes:

    - ``"sequential"`` threads the output of one child as the context of the
      next; the final child's ``(output, alpha)`` is returned.
    - ``"parallel"`` invokes each child on the same context and averages the
      outputs and alphas. Averaged alpha is clamped to ``min=1e-6`` to
      preserve the strict-positivity invariant.
    """

    backing_type = "composite"

    def __init__(self, children: list[Schema], mode: str) -> None:
        if mode not in {"sequential", "parallel"}:
            raise ValueError(
                f"Unknown composite mode: {mode!r}. "
                "Expected 'sequential' or 'parallel'."
            )
        self.children = children
        self.mode = mode

    def forward(
        self, context: Tensor, slot_values: dict[str, Any] | None = None
    ) -> tuple[Tensor, Tensor]:
        if self.mode == "sequential":
            current = context
            alpha = torch.ones_like(context)
            for child in self.children:
                current, alpha = child.predict(current, slot_values)
            return current, alpha

        outputs: list[Tensor] = []
        alphas: list[Tensor] = []
        for child in self.children:
            output, alpha = child.predict(context, slot_values)
            outputs.append(output)
            alphas.append(alpha)
        mean_output = torch.stack(outputs, dim=0).mean(dim=0)
        mean_alpha = torch.stack(alphas, dim=0).mean(dim=0).clamp(min=1e-6)
        return mean_output, mean_alpha
