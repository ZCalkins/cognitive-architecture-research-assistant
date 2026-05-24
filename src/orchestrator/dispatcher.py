"""Dispatcher — selects and invokes schemas for a query.

Strange-loop role
-----------------
The dispatcher is where action selection lives. In Session 1 it is plain
top-k by activation. The self-schema (Phase 3) will modulate this selection
(downward causation), and active inference (Session 2) will replace top-k with
expected-free-energy minimization. Keeping selection in the orchestrator (not
in a specialist) is the (C) firewall: specialists are oracles, never deciders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

    from src.instrumentation.probes import Probe
    from src.orchestrator.registry import SchemaRegistry
    from src.orchestrator.workspace import Workspace
    from src.schemas.base import Schema


class Dispatcher:
    """Selects active schemas by activation and invokes their predictions."""

    def __init__(self, registry: SchemaRegistry, workspace: Workspace) -> None:
        self.registry = registry
        self.workspace = workspace

    def dispatch(
        self,
        query_context: Tensor,
        slot_values: dict[str, Any] | None = None,
        top_k: int = 3,
        self_schema_state: Tensor | None = None,
        probe: Probe | None = None,
    ) -> list[tuple[Schema, Tensor, Tensor]]:
        """Select top-k active schemas and invoke each.

        TODO(session2): replace top-k with active-inference action selection
        over expected free energy.
        """
        active = self.registry.active(query_context, threshold=float("-inf"))
        selected = active[:top_k]
        self.workspace.assemble_context(
            query_context, selected, self_schema_state, probe
        )

        results: list[tuple[Schema, Tensor, Tensor]] = []
        for schema in selected:
            output, alpha = schema.predict(query_context, slot_values)
            results.append((schema, output, alpha))
            if probe is not None:
                schema_id = str(schema.meta.schema_id)
                probe.record(f"output:{schema_id}", output)
                probe.record(f"alpha:{schema_id}", alpha)

        if probe is not None:
            probe.record_scalar("n_dispatched", len(results))
        return results
