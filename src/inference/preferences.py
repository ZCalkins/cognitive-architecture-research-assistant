"""Preferences — the pragmatic-value term for active-inference action selection.

Strange-loop role
-----------------
Preferences encode which outcomes the orchestrator favors when choosing which
specialist to invoke. In Session 2 they are hand-coded heuristics; from
Session 3 they become learnable from engagement signals. The ``score``
interface is the load-bearing part — the active-inference selector depends only
on it, so the heuristic can be swapped for a learned model without touching the
selector.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor


class Preferences(ABC):
    """Scores a predicted ``(output, alpha)``; higher means more preferred."""

    @abstractmethod
    def score(self, output: Tensor, alpha: Tensor) -> float:
        ...


class ConfidencePreferences(Preferences):
    """Session 2 default: prefer higher confidence (higher mean Dirichlet alpha)."""

    def score(self, output: Tensor, alpha: Tensor) -> float:
        return float(alpha.sum() / alpha.numel())


class UniformPreferences(Preferences):
    """Indifferent baseline (always 0.0) — the pragmatic-ablation condition."""

    def score(self, output: Tensor, alpha: Tensor) -> float:
        return 0.0
