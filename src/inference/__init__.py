"""Active inference — the orchestrator's action-selection mechanism.

Session 2 deliverable (ARCHITECTURE.md "Phase 1 deliverable 8"). Expected
free energy = pragmatic value (preferences) + epistemic value (predicted
uncertainty) over candidate specialist invocations. The self-schema (Phase 3)
will modulate the generative model via the hypernetwork — downward causation.
"""

from src.inference.active_inference import ActiveInferenceSelector
from src.inference.generative_model import GenerativeModel
from src.inference.preferences import (
    ConfidencePreferences,
    Preferences,
    UniformPreferences,
)

__all__ = [
    "ActiveInferenceSelector",
    "ConfidencePreferences",
    "GenerativeModel",
    "Preferences",
    "UniformPreferences",
]
