"""Piagetian lifecycle — the schema population reorganizing itself.

Strange-loop role
-----------------
This is the mechanism by which the schema population spawns / splits / merges /
prunes in response to its own prediction errors — the substrate for (C)
self-organizing specialist emergence. Session 2 defines the API and wires the
operations that Session-1 primitives already support (``compose`` via
``Schema.compose``; ``prune`` via ``registry.unregister``). The actual trigger
*policies* (thresholds, buffering, rate limits) are a Phase 2 deliverable; the
triggers here are stubs returning no events.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

if TYPE_CHECKING:
    from torch import Tensor

    from src.orchestrator.registry import SchemaRegistry
    from src.persistence.store import SQLiteStore
    from src.schemas.base import Schema


@dataclass
class LifecycleEvent:
    """A proposed (or applied) lifecycle event with full provenance."""

    event_type: str
    affected_schema_ids: list[UUID]
    trigger_description: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


class LifecycleTrigger(ABC):
    """Evaluates the schema population and proposes lifecycle events."""

    trigger_type: str = "abstract"

    @abstractmethod
    def evaluate(self, schemas: list[Schema], context: Tensor) -> list[LifecycleEvent]:
        """Return proposed events; empty list if no trigger fires."""
        ...


class SpawnTrigger(LifecycleTrigger):
    """Proposes spawning a new specialist after sustained localized error.

    Phase 2 policy: when N consecutive prediction errors exceed
    ``error_threshold`` in a localized region of input space (buffered against
    single anomalies, rate-limited), propose a spawn event. Session 2 stub:
    returns [].
    """

    trigger_type = "spawn"

    def __init__(self, error_threshold: float = 0.5, buffer_size: int = 5) -> None:
        self.error_threshold = error_threshold
        self.buffer_size = buffer_size
        self._error_buffer: list[tuple[Tensor, float]] = []

    def evaluate(self, schemas: list[Schema], context: Tensor) -> list[LifecycleEvent]:
        return []


class SplitTrigger(LifecycleTrigger):
    """Proposes splitting a schema whose predictions are bimodal.

    Phase 2 policy: detect sustained bimodality in a schema's confidence
    trajectory and propose a split. Session 2 stub: returns [].
    """

    trigger_type = "split"

    def __init__(self, bimodality_threshold: float = 0.7, window_size: int = 50) -> None:
        self.bimodality_threshold = bimodality_threshold
        self.window_size = window_size

    def evaluate(self, schemas: list[Schema], context: Tensor) -> list[LifecycleEvent]:
        return []


class MergeTrigger(LifecycleTrigger):
    """Proposes merging two schemas with high mutual prediction similarity.

    Phase 2 policy: detect sustained high mutual prediction similarity between
    two schemas and propose a merge. Session 2 stub: returns [].
    """

    trigger_type = "merge"

    def __init__(self, similarity_threshold: float = 0.9, window_size: int = 50) -> None:
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size

    def evaluate(self, schemas: list[Schema], context: Tensor) -> list[LifecycleEvent]:
        return []


class PruneTrigger(LifecycleTrigger):
    """Proposes pruning schemas with extended low activation + low utility.

    Phase 2 policy: identify schemas idle beyond ``idle_steps`` with activation
    below ``activation_threshold`` and no recent positive engagement signal.
    Session 2 stub: returns [].
    """

    trigger_type = "prune"

    def __init__(self, activation_threshold: float = 0.05, idle_steps: int = 100) -> None:
        self.activation_threshold = activation_threshold
        self.idle_steps = idle_steps

    def evaluate(self, schemas: list[Schema], context: Tensor) -> list[LifecycleEvent]:
        return []


class LifecycleManager:
    """Evaluates triggers and applies the events they propose.

    Session 2 wires ``compose`` (via ``Schema.compose``) and ``prune`` (via
    ``registry.unregister``); ``spawn`` / ``split`` / ``merge`` raise
    ``NotImplementedError`` pointing to Phase 2.
    """

    def __init__(
        self,
        triggers: list[LifecycleTrigger],
        registry: SchemaRegistry,
        store: SQLiteStore,
    ) -> None:
        self.triggers = triggers
        self.registry = registry
        self.store = store

    def evaluate(self, context: Tensor) -> list[LifecycleEvent]:
        events: list[LifecycleEvent] = []
        for trigger in self.triggers:
            events.extend(trigger.evaluate(self.registry.all(), context))
        return events

    def apply(self, events: list[LifecycleEvent]) -> None:
        for event in events:
            if event.event_type == "prune":
                for schema_id in event.affected_schema_ids:
                    self.registry.unregister(schema_id)
            elif event.event_type == "compose":
                self._apply_compose(event)
            elif event.event_type in {"spawn", "split", "merge"}:
                raise NotImplementedError(
                    f"{event.event_type.capitalize()} dynamics are a Phase 2 "
                    "deliverable. Session 2 only defines the API."
                )
            else:
                raise ValueError(f"Unknown lifecycle event_type: {event.event_type!r}")
            self.store.record_lifecycle_event(
                schema_id=",".join(str(sid) for sid in event.affected_schema_ids),
                event_type=event.event_type,
                trigger={"description": event.trigger_description},
                provenance={"metadata": event.metadata, "timestamp": event.timestamp},
            )

    def _apply_compose(self, event: LifecycleEvent) -> None:
        if len(event.affected_schema_ids) != 2:
            raise ValueError("compose event requires exactly two affected schema ids.")
        parent = self.registry.get(event.affected_schema_ids[0])
        other = self.registry.get(event.affected_schema_ids[1])
        if parent is None or other is None:
            raise ValueError("compose event references unknown schema id(s).")
        composite = parent.compose(other, mode=event.metadata.get("mode", "sequential"))
        self.registry.register(composite)

    def step(self, context: Tensor) -> list[LifecycleEvent]:
        events = self.evaluate(context)
        self.apply(events)
        return events
