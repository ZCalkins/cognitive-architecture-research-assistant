"""Orchestrator loop and interactive session — the minimal end-to-end cycle.

Strange-loop role
-----------------
This is a deliberately minimal skeleton. The self-schema, hypernetwork, active
inference, and Piagetian lifecycle all hook in later sessions. The current
loop exists to verify end-to-end integration: a query enters, schemas are
selected and invoked, the event is persisted, and the workspace is updated.
The orchestrator — not any LLM — owns this cycle (the (C) firewall).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

    from src.instrumentation.probes import Probe
    from src.orchestrator.dispatcher import Dispatcher
    from src.orchestrator.registry import SchemaRegistry
    from src.orchestrator.workspace import Workspace
    from src.persistence.store import SQLiteStore


class Orchestrator:
    """The multi-agent meta-coordinator — the strange-loop research subject."""

    def __init__(
        self,
        registry: SchemaRegistry,
        workspace: Workspace,
        dispatcher: Dispatcher,
        store: SQLiteStore,
        step_count: int = 0,
    ) -> None:
        self.registry = registry
        self.workspace = workspace
        self.dispatcher = dispatcher
        self.store = store
        self.step_count = step_count

    def step(
        self,
        query_context: Tensor,
        slot_values: dict[str, Any] | None = None,
        probe: Probe | None = None,
    ) -> dict:
        """Advance the orchestrator one cycle and persist the event."""
        self.step_count += 1
        results = self.dispatcher.dispatch(
            query_context, slot_values=slot_values, probe=probe
        )

        if results:
            alpha_sums = [float(alpha.detach().sum(-1).mean()) for _, _, alpha in results]
            mean_alpha_sum = sum(alpha_sums) / len(alpha_sums)
        else:
            mean_alpha_sum = 0.0

        summary = {
            "step": self.step_count,
            "n_dispatched": len(results),
            "schema_ids": [str(schema.meta.schema_id) for schema, _, _ in results],
            "mean_alpha_sum": mean_alpha_sum,
        }
        self.store.record_orchestrator_event("step", summary)
        self.workspace.broadcast({"step": self.step_count, "summary": summary})
        return summary

    def attach_session(self) -> Session:
        """Open an interactive session bound to this orchestrator."""
        return Session(self)


class Session:
    """Interactive session context manager.

    Records session-open and session-close events. ``__enter__`` is
    future-proofed for state restoration from the store (Session 1 only
    records the open event).
    """

    def __init__(self, orchestrator: Orchestrator) -> None:
        self.orchestrator = orchestrator

    def __enter__(self) -> Orchestrator:
        self.orchestrator.store.record_orchestrator_event("session_open", {})
        return self.orchestrator

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        self.orchestrator.store.record_orchestrator_event(
            "session_close", {"final_step": self.orchestrator.step_count}
        )
        return False
