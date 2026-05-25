"""Active inference — action selection by expected-free-energy minimization.

Strange-loop role
-----------------
This is the orchestrator's action-selection mechanism: it scores candidate
specialist invocations by pragmatic value (preference satisfaction) plus
epistemic value (uncertainty to be resolved). In Phase 3 the self-schema's
state will MODULATE the generative model via the hypernetwork — that is where
downward causation enters and the strange loop closes. For Session 2 the
generative model stands alone, but the selector's API is designed so that
modulation slots in cleanly later.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

    from src.inference.generative_model import GenerativeModel
    from src.inference.preferences import Preferences
    from src.instrumentation.probes import Probe
    from src.schemas.base import Schema


class ActiveInferenceSelector:
    """Selects specialists by expected free energy (pragmatic + epistemic value).

    Parameters
    ----------
    generative_model:
        Predicts each candidate specialist's output/alpha/uncertainty.
    preferences:
        Pragmatic-value scorer over the predicted output/alpha.
    epistemic_weight:
        Weight on epistemic value (predicted uncertainty = expected info gain).
    pragmatic_weight:
        Weight on pragmatic value (preference satisfaction).
    """

    def __init__(
        self,
        generative_model: GenerativeModel,
        preferences: Preferences,
        epistemic_weight: float = 0.5,
        pragmatic_weight: float = 1.0,
    ) -> None:
        self.generative_model = generative_model
        self.preferences = preferences
        self.epistemic_weight = epistemic_weight
        self.pragmatic_weight = pragmatic_weight

    def score_action(self, schema: Schema, context: Tensor) -> dict:
        """Score a single candidate; returns pragmatic/epistemic/total and predictions."""
        with torch.no_grad():
            predicted_output, predicted_alpha, predicted_uncertainty = (
                self.generative_model.predict(schema, context)
            )
        pragmatic = self.preferences.score(predicted_output, predicted_alpha)
        epistemic = float(predicted_uncertainty)
        total = self.pragmatic_weight * pragmatic + self.epistemic_weight * epistemic
        return {
            "schema_id": schema.meta.schema_id,
            "pragmatic": pragmatic,
            "epistemic": epistemic,
            "total": total,
            "predicted_output": predicted_output,
            "predicted_alpha": predicted_alpha,
            "predicted_uncertainty": epistemic,
        }

    def select(
        self,
        schemas: list[Schema],
        context: Tensor,
        k: int = 3,
        probe: Probe | None = None,
    ) -> list[tuple[Schema, dict]]:
        """Score all candidates and return the top-k ``(schema, score_dict)``."""
        scored = [(schema, self.score_action(schema, context)) for schema in schemas]
        scored.sort(key=lambda pair: pair[1]["total"], reverse=True)
        if probe is not None:
            for schema, score in scored:
                short = str(schema.meta.schema_id)[:8]
                probe.record_scalar(f"pragmatic:{short}", score["pragmatic"])
                probe.record_scalar(f"epistemic:{short}", score["epistemic"])
                probe.record_scalar(f"total:{short}", score["total"])
        return scored[:k]
