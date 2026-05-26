"""Global workspace — the context-construction service.

Strange-loop role
-----------------
This REPLACES the original "broadcast layer" framing (ARCHITECTURE.md "Global
workspace"). When a specialist is invoked, the workspace assembles its context
from the active schemas, the self-schema state, recent workspace contents, and
(later) KB retrievals.

The self-schema's privilege is enforced *architecturally, not
philosophically*: ``self_schema_state`` is ALWAYS a key of the assembled
context dict, even when its value is ``None``. That always-keyed slot is the
(C) firewall expressed at the type level — there is nowhere for object-level
context to be assembled without a place reserved for the self-schema.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

    from src.instrumentation.probes import Probe
    from src.orchestrator.registry import SchemaRegistry
    from src.schemas.base import Schema


class Workspace:
    """Assembles per-invocation context from active schemas and self-state."""

    def __init__(self, registry: SchemaRegistry, rolling_window_size: int = 16) -> None:
        self.registry = registry
        self.rolling_window_size = rolling_window_size
        self._recent: list[dict] = []

    def assemble_context(
        self,
        query_context: Tensor,
        active_schemas: list[Schema],
        self_schema_state: Tensor | None = None,
        probe: Probe | None = None,
    ) -> dict:
        """Assemble a context bundle for a specialist invocation.

        The returned dict always contains a ``self_schema_state`` key (the (C)
        firewall), even when no self-schema exists yet (Session 1).
        """
        active_schema_states = [
            (schema.meta.schema_id, schema.embedding) for schema in active_schemas
        ]
        context = {
            "query_context": query_context,
            "active_schema_states": active_schema_states,
            "self_schema_state": self_schema_state,
            "recent_workspace": list(self._recent),
        }
        if probe is not None:
            probe.record("query_context", query_context)
            probe.record_scalar("n_active_schemas", len(active_schemas))
        return context

    def broadcast(self, content: dict) -> None:
        """Append ``content`` to the rolling window, trimming to size."""
        self._recent.append(content)
        if len(self._recent) > self.rolling_window_size:
            self._recent = self._recent[-self.rolling_window_size :]
