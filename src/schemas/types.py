"""Shared dataclasses for the polymorphic schema framework.

Three small value types underpin the schema interface:

- ``TypedSlot`` — Minsky-frame slots. Slots are downstream-referenceable and
  are **the substrate for Phase 3 downward causation**: the hypernetwork
  modulates schema behavior by writing typed values into slots.
- ``SpawnRecord`` — provenance for every schema. Provenance is **required by
  Phase 5 criterion (D)**: every architectural claim must be inspectable, so
  every schema records who/what/when spawned it and in response to what error.
- ``StructureMapping`` — the result of an analogical mapping between schemas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID


@dataclass
class TypedSlot:
    """A named, typed slot on a schema (a Minsky-frame role).

    Slots are the substrate for Phase 3 downward causation: the hypernetwork
    writes typed modulations into slots to steer object-level behavior.

    Parameters
    ----------
    name:
        Slot identifier, referenceable by downstream schemas.
    type_hint:
        A string type tag for now. Could become a runtime type later.
    description:
        Human-readable description of the slot's role.
    default:
        Default value when the slot is unfilled.
    """

    name: str
    type_hint: str
    description: str = ""
    default: Any = None


@dataclass
class SpawnRecord:
    """Provenance of a schema — required by Phase 5 criterion (D).

    Every schema records how it came to exist so that lifecycle dynamics and
    downward-causal claims remain inspectable and falsifiable.

    Parameters
    ----------
    parent_ids:
        Schema ids that gave rise to this schema (empty for seed schemas).
    created_at_step:
        Orchestrator step at which the schema was created.
    event:
        Lifecycle event that created it: "spawn", "specialize", "split",
        "merge", "compose".
    trigger_description:
        Free-text description of the triggering condition (e.g. the
        prediction-error pattern that prompted a spawn).
    """

    parent_ids: tuple[UUID, ...] = ()
    created_at_step: int = 0
    event: str = ""
    trigger_description: str = ""


@dataclass
class StructureMapping:
    """An analogical mapping between two schemas.

    Session 1 implementation is naive slot-name-matching. Real Gentner SMT
    (relation-preserving, systematicity-biased) is a Session 2+ deliverable.

    Parameters
    ----------
    source_schema_id:
        Id of the schema the mapping is from.
    target_schema_id:
        Id of the schema the mapping is to.
    slot_correspondences:
        List of (source_slot_name, target_slot_name) pairs.
    confidence:
        Confidence in the mapping, in [0, 1].
    """

    source_schema_id: UUID
    target_schema_id: UUID
    slot_correspondences: list[tuple[str, str]]
    confidence: float
