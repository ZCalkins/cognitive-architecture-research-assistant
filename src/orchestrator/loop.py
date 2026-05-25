"""Orchestrator loop and interactive session — the end-to-end cycle.

Strange-loop role
-----------------
The orchestrator owns the cognitive cycle (the (C) firewall): a query enters,
schemas are selected (active inference if a selector is wired, else the top-k
baseline) and invoked, the generative model learns from each observation, the
lifecycle manager may reorganize the population, the event is persisted, and
the workspace is updated. The self-schema and hypernetwork (Phase 3) hook into
this same loop later — they do not replace it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

    from src.inference.active_inference import ActiveInferenceSelector
    from src.inference.generative_model import GenerativeModel
    from src.instrumentation.probes import Probe
    from src.orchestrator.dispatcher import Dispatcher
    from src.orchestrator.registry import SchemaRegistry
    from src.orchestrator.workspace import Workspace
    from src.persistence.store import SQLiteStore
    from src.schemas.lifecycle import LifecycleManager


class Orchestrator:
    """The multi-agent meta-coordinator — the strange-loop research subject."""

    def __init__(
        self,
        registry: SchemaRegistry,
        workspace: Workspace,
        dispatcher: Dispatcher,
        store: SQLiteStore,
        step_count: int = 0,
        selector: ActiveInferenceSelector | None = None,
        generative_model: GenerativeModel | None = None,
        lifecycle_manager: LifecycleManager | None = None,
    ) -> None:
        if selector is not None and generative_model is None:
            raise ValueError(
                "ActiveInferenceSelector requires a GenerativeModel for learning "
                "updates. Pass generative_model alongside selector."
            )
        self.registry = registry
        self.workspace = workspace
        self.dispatcher = dispatcher
        self.store = store
        self.step_count = step_count
        self.selector = selector
        self.generative_model = generative_model
        self.lifecycle_manager = lifecycle_manager

        if selector is not None:
            self.dispatcher.selector = selector
        if generative_model is not None:
            self.dispatcher.on_observation = (
                lambda schema, ctx, output, alpha: generative_model.update(
                    schema, ctx, output, alpha
                )
            )

    def step(
        self,
        query_context: Tensor,
        slot_values: dict[str, Any] | None = None,
        probe: Probe | None = None,
    ) -> dict:
        """Advance the orchestrator one cycle and persist the event.

        Learning updates happen inside ``dispatch`` via the ``on_observation``
        callback wired in ``__init__``; the lifecycle manager (if any) runs
        after dispatch.
        """
        self.step_count += 1
        results = self.dispatcher.dispatch(
            query_context, slot_values=slot_values, probe=probe
        )

        if results:
            alpha_sums = [float(alpha.detach().sum(-1).mean()) for _, _, alpha in results]
            mean_alpha_sum = sum(alpha_sums) / len(alpha_sums)
        else:
            mean_alpha_sum = 0.0

        lifecycle_events = (
            self.lifecycle_manager.step(query_context)
            if self.lifecycle_manager is not None
            else []
        )

        summary = {
            "step": self.step_count,
            "n_dispatched": len(results),
            "schema_ids": [str(schema.meta.schema_id) for schema, _, _ in results],
            "mean_alpha_sum": mean_alpha_sum,
            "selection_mode": 1.0 if self.selector is not None else 0.0,
            "n_lifecycle_events": len(lifecycle_events),
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
