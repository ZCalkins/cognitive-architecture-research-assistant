"""Schema registry — the orchestrator's population of schemas.

Strange-loop role
-----------------
The registry is the population the self-schema (Phase 3) models and the
lifecycle (Phase 2) reorganizes. Registration/unregistration are the
substrate for spawn/prune. ``active`` is the substrate for context-sensitive
routing, which the self-schema will later modulate (downward causation).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from torch import Tensor

    from src.schemas.base import Schema


class SchemaRegistry:
    """An indexed population of schemas keyed by schema id."""

    def __init__(self) -> None:
        self._schemas: dict[UUID, Schema] = {}

    def register(self, schema: Schema) -> UUID:
        """Add ``schema`` to the population; returns its id."""
        self._schemas[schema.meta.schema_id] = schema
        return schema.meta.schema_id

    def unregister(self, schema_id: UUID) -> None:
        """Remove a schema (the Phase 2 prune primitive)."""
        self._schemas.pop(schema_id, None)

    def get(self, schema_id: UUID) -> Schema | None:
        return self._schemas.get(schema_id)

    def all(self) -> list[Schema]:
        return list(self._schemas.values())

    def active(self, context: Tensor, threshold: float = 0.0) -> list[Schema]:
        """Schemas activating in ``context``, sorted by activation desc.

        Returns schemas whose ``activate_in(context) >= threshold``.
        """
        scored = [(schema, schema.activate_in(context)) for schema in self._schemas.values()]
        scored = [(schema, score) for schema, score in scored if score >= threshold]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [schema for schema, _ in scored]

    def __iter__(self) -> Iterator[Schema]:
        return iter(self._schemas.values())

    def __len__(self) -> int:
        return len(self._schemas)

    def __contains__(self, schema_id: UUID) -> bool:
        return schema_id in self._schemas
