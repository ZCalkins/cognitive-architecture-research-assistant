"""The ``Schema`` — the universal unit of the architecture.

Each schema is a ``(metadata, content, interface)`` triple: ``SchemaMeta``
(referent, hierarchy, provenance, calibration), a learnable ``embedding`` and
polymorphic ``backing`` (content), and the predict / map / specialize /
generalize / compose / activate / ablate interface.

Strange-loop role
-----------------
There is no special "self" type. The self-schema (Phase 3 deliverable) is
just a ``Schema`` whose ``referent`` is the orchestrator's own state, and the
meta-self-schema is the self-schema taking another ``Schema`` (itself) as
input — **no special-casing anywhere**. The strange loop is realized by the
self-schema's modulations flowing back through the same uniform interface that
every object schema exposes. Slots, provenance, and ablation hooks exist so
the downward-causal claim is measurable (Phase 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import torch
from torch import Tensor, nn

from src.schemas.backings import CompositeBacking, SchemaBacking
from src.schemas.types import SpawnRecord, StructureMapping, TypedSlot


@dataclass
class SchemaMeta:
    """Metadata for a schema: identity, hierarchy, provenance, calibration."""

    schema_id: UUID = field(default_factory=uuid4)
    referent: str = ""
    parents: list[Schema] = field(default_factory=list)
    children: list[Schema] = field(default_factory=list)
    provenance: SpawnRecord = field(default_factory=SpawnRecord)
    created_at_step: int = 0
    last_active_step: int = 0
    activation_count: int = 0
    confidence_trajectory: list[float] = field(default_factory=list)


class Schema(nn.Module):
    """A parameterized module representing some referent.

    Polymorphic across backings (neural / llm / symbolic / composite). The
    same class is used for object schemas and (in Phase 3) the self-schema.

    Parameters
    ----------
    referent:
        What this schema is about.
    backing:
        Polymorphic implementation (see ``backings``).
    latent_dim:
        Dimensionality of the latent semantic content and context signature.
    slots:
        Minsky-frame slots; substrate for hypernetwork modulation (Phase 3).
    context_signature:
        Vector describing which contexts activate this schema. Defaults to
        zeros (never activates until learned/assigned).
    parents:
        Parent schemas in the generalization hierarchy.
    parent_event:
        Lifecycle event that created this schema relative to its parents.
    """

    def __init__(
        self,
        referent: str,
        backing: SchemaBacking,
        latent_dim: int,
        slots: dict[str, TypedSlot] | None = None,
        context_signature: Tensor | None = None,
        parents: list[Schema] | None = None,
        parent_event: str = "",
    ) -> None:
        super().__init__()
        self.backing = backing
        self.latent_dim = latent_dim
        self.slots: dict[str, TypedSlot] = slots if slots is not None else {}

        parent_list = list(parents) if parents else []
        parent_ids = tuple(p.meta.schema_id for p in parent_list)
        self.meta = SchemaMeta(
            referent=referent,
            parents=parent_list,
            provenance=SpawnRecord(parent_ids=parent_ids, event=parent_event),
        )

        self.embedding = nn.Parameter(torch.zeros(latent_dim))

        if context_signature is None:
            context_signature = torch.zeros(latent_dim)
        self.register_buffer("context_signature", context_signature.clone())

        backing_module = getattr(backing, "module", None)
        if isinstance(backing_module, nn.Module):
            self._backing_module = backing_module

        self._ablated: bool = False

    def predict(
        self, context: Tensor, slot_values: dict[str, Any] | None = None
    ) -> tuple[Tensor, Tensor]:
        """Predict ``(output, dirichlet_alpha)`` for ``context``.

        When ablated, returns a zero output and a flat (all-ones) Dirichlet,
        and does not count as an activation.
        """
        if self._ablated:
            batch = context.shape[0] if context.dim() > 1 else 1
            zeros = torch.zeros(batch, self.latent_dim)
            ones = torch.ones(batch, self.latent_dim)
            return zeros, ones

        self.meta.activation_count += 1
        output, alpha = self.backing.forward(context, slot_values)
        with torch.no_grad():
            alpha_sum = alpha.sum(-1)
            confidence = (alpha_sum / (alpha_sum + alpha.shape[-1])).mean()
        self.meta.confidence_trajectory.append(float(confidence))
        return output, alpha

    def forward(
        self, context: Tensor, slot_values: dict[str, Any] | None = None
    ) -> tuple[Tensor, Tensor]:
        return self.predict(context, slot_values)

    def activate_in(self, context: Tensor) -> float:
        """Context-relevance: cosine similarity to the context signature.

        Multi-batch contexts are mean-pooled first. A zero context signature
        (the default) yields 0.0.
        """
        ctx = context.mean(dim=0) if context.dim() > 1 else context
        sig = self.context_signature
        denom = (ctx.norm() * sig.norm()).clamp(min=1e-8)
        return float(torch.dot(ctx, sig) / denom)

    def map_to(self, target: Schema) -> StructureMapping:
        """Analogical mapping to ``target``.

        TODO(session2): replace with Gentner SMT (preserve relations,
        systematicity bias). Session 1 matches slots by name equality.
        """
        correspondences = [(name, name) for name in self.slots if name in target.slots]
        confidence = (
            len(correspondences) / max(len(self.slots), 1) if self.slots else 0.0
        )
        return StructureMapping(
            source_schema_id=self.meta.schema_id,
            target_schema_id=target.meta.schema_id,
            slot_correspondences=correspondences,
            confidence=confidence,
        )

    def specialize(self, constraint: dict[str, Any]) -> Schema:
        """Create a specialization child under a constraint.

        Shares the backing for v0 (deep-copy logic comes later). Records the
        constraint in the child's provenance and links the child into the
        children hierarchy.
        """
        child = Schema(
            referent=self.meta.referent,
            backing=self.backing,
            latent_dim=self.latent_dim,
            slots=dict(self.slots),
            context_signature=self.context_signature.clone(),
            parents=[self],
            parent_event="specialize",
        )
        child.meta.provenance.trigger_description = str(constraint)
        self.meta.children.append(child)
        return child

    def generalize(self) -> Schema:
        """Return the first parent, if any.

        Standalone generalization (structural lifting without an existing
        parent) is a Phase 2 deliverable.
        """
        if self.meta.parents:
            return self.meta.parents[0]
        raise NotImplementedError(
            "Standalone generalize() requires structural lifting; "
            "Phase 2 deliverable."
        )

    def compose(self, other: Schema, mode: str = "sequential") -> Schema:
        """Compose with ``other`` into a new composite schema."""
        backing = CompositeBacking([self, other], mode)
        combined_slots = {**self.slots, **other.slots}
        return Schema(
            referent=f"compose({self.meta.referent}, {other.meta.referent})",
            backing=backing,
            latent_dim=self.latent_dim,
            slots=combined_slots,
            parents=[self, other],
            parent_event="compose",
        )

    def ablate(self) -> None:
        """Disable the schema (Phase 5 ablation studies)."""
        self._ablated = True

    def restore(self) -> None:
        """Re-enable a previously ablated schema."""
        self._ablated = False

    @property
    def is_ablated(self) -> bool:
        return self._ablated

    def __repr__(self) -> str:
        short_id = str(self.meta.schema_id)[:8]
        return (
            f"Schema(referent={self.meta.referent!r}, "
            f"backing={self.backing.backing_type}, "
            f"id={short_id}, ablated={self._ablated})"
        )
