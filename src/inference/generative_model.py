"""Generative model — the orchestrator's predictive model of specialist behavior.

Strange-loop role
-----------------
This is the orchestrator's learned model of *which specialist does what in
which context* — a model of its own object-level behavior. In Phase 3 the
self-schema will be conditioned on (and the hypernetwork will modulate) this
generative model, closing the strange loop. For Session 2 it stands alone: a
small, few-shot-learnable predictor that drives active-inference action
selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn.functional import mse_loss, softplus

if TYPE_CHECKING:
    from src.schemas.base import Schema


class GenerativeModel(nn.Module):
    """Predicts per-specialist ``(output, alpha, uncertainty)`` from context.

    Parameters
    ----------
    latent_dim:
        Dimensionality of schema outputs and the query context.
    schema_dim:
        Dimensionality of schema context-signature vectors. Defaults to
        ``latent_dim``.
    hidden_dim:
        Hidden width of the MLP.
    """

    def __init__(
        self, latent_dim: int, schema_dim: int | None = None, hidden_dim: int = 64
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.schema_dim = schema_dim if schema_dim is not None else latent_dim
        self.trunk = nn.Sequential(
            nn.Linear(self.schema_dim + latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.output_head = nn.Linear(hidden_dim, latent_dim)
        self.log_alpha_head = nn.Linear(hidden_dim, latent_dim)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)
        self._optim = torch.optim.Adam(self.parameters(), lr=1e-3)

    def _features(self, schema: Schema, context: Tensor) -> Tensor:
        ctx = context.mean(dim=0) if context.dim() > 1 else context
        return torch.cat([schema.context_signature, ctx], dim=-1)

    def predict(self, schema: Schema, context: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(predicted_output, predicted_alpha, predicted_uncertainty)``.

        ``predicted_alpha`` is ``exp(.)`` of a head output (strictly positive);
        ``predicted_uncertainty`` is a softplus scalar (non-negative).
        """
        hidden = self.trunk(self._features(schema, context))
        predicted_output = self.output_head(hidden)
        predicted_alpha = torch.exp(self.log_alpha_head(hidden))
        predicted_uncertainty = softplus(self.uncertainty_head(hidden)).squeeze(-1)
        return predicted_output, predicted_alpha, predicted_uncertainty

    def forward(self, schema: Schema, context: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.predict(schema, context)

    def update(
        self,
        schema: Schema,
        context: Tensor,
        observed_output: Tensor,
        observed_alpha: Tensor,
        lr: float = 1e-3,
    ) -> float:
        """One gradient step toward the observed (output, alpha). Returns loss."""
        for group in self._optim.param_groups:
            group["lr"] = lr
        observed_output = (
            observed_output.mean(dim=0) if observed_output.dim() > 1 else observed_output
        )
        observed_alpha = (
            observed_alpha.mean(dim=0) if observed_alpha.dim() > 1 else observed_alpha
        )
        self._optim.zero_grad()
        predicted_output, predicted_alpha, _ = self.predict(schema, context)
        loss = mse_loss(predicted_output, observed_output.detach()) + mse_loss(
            predicted_alpha, observed_alpha.detach()
        )
        loss.backward()
        self._optim.step()
        return float(loss.detach())
