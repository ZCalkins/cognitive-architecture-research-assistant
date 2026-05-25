"""Dispatcher — selects and invokes schemas for a query.

Strange-loop role
-----------------
The dispatcher is where action selection lives. With an
``ActiveInferenceSelector`` wired in, selection is by expected free energy
(pragmatic + epistemic value); without one, it falls back to top-k by
activation. Keeping selection in the orchestrator (not in a specialist) is the
(C) firewall: specialists are oracles, never deciders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from src.inference.active_inference import ActiveInferenceSelector
    from src.instrumentation.probes import Probe
    from src.orchestrator.registry import SchemaRegistry
    from src.orchestrator.workspace import Workspace
    from src.schemas.base import Schema


class Dispatcher:
    """Selects schemas (active inference or top-k baseline) and invokes them."""

    def __init__(
        self,
        registry: SchemaRegistry,
        workspace: Workspace,
        selector: ActiveInferenceSelector | None = None,
        on_observation: Callable[[Schema, Tensor, Tensor, Tensor], None] | None = None,
    ) -> None:
        self.registry = registry
        self.workspace = workspace
        self.selector = selector
        self.on_observation = on_observation

    def dispatch(
        self,
        query_context: Tensor,
        slot_values: dict[str, Any] | None = None,
        top_k: int = 3,
        self_schema_state: Tensor | None = None,
        probe: Probe | None = None,
    ) -> list[tuple[Schema, Tensor, Tensor]]:
        """Select schemas and invoke each.

        NOTE: the top-k-by-activation path (``selector is None``) is preserved
        deliberately as the ablation control for Phase 5 criterion 2 (downward
        causation traceability). Do not remove it — without the baseline, the
        active-inference selector's effect cannot be measured.
        """
        active = self.registry.active(query_context, threshold=float("-inf"))

        if self.selector is not None:
            selected = [schema for schema, _ in self.selector.select(
                active, query_context, k=top_k, probe=probe
            )]
            selection_mode = 1.0
        else:
            selected = active[:top_k]
            selection_mode = 0.0

        self.workspace.assemble_context(
            query_context, selected, self_schema_state, probe
        )

        results: list[tuple[Schema, Tensor, Tensor]] = []
        for schema in selected:
            output, alpha = schema.predict(query_context, slot_values)
            results.append((schema, output, alpha))
            if self.on_observation is not None:
                self.on_observation(schema, query_context, output, alpha)
            if probe is not None:
                schema_id = str(schema.meta.schema_id)
                probe.record(f"output:{schema_id}", output)
                probe.record(f"alpha:{schema_id}", alpha)

        if probe is not None:
            probe.record_scalar("n_dispatched", len(results))
            probe.record_scalar("selection_mode", selection_mode)
        return results
