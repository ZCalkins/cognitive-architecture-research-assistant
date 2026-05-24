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

    Session 1 only defines the *shape* of an LLM-backed schema. The actual
    call path lands in Session 2 alongside active-inference-mediated
    invocation, where the provider is wired through the orchestrator.
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
        if slot_values is None or "_llm_provider" not in slot_values:
            raise NotImplementedError(
                "LLMBacking.forward requires an LLMProvider in "
                "slot_values['_llm_provider']. Wire through the orchestrator. "
                "Session 2 deliverable: active-inference mediated invocation."
            )
        raise NotImplementedError("LLM call path is a Session 2 deliverable.")


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
