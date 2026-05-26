from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LossWeights:
    task: float = 1.0
    causal: float = 0.02
    energy: float = 0.02
    uncertainty: float = 0.01
    consolidation: float = 1.0


@dataclass
class LossBreakdown:
    total: float
    task: float
    causal: float
    energy: float
    uncertainty: float
    consolidation: float


class CausperaLoss(nn.Module):
    """Unified training objective for Causpera."""

    def __init__(
        self,
        weights: LossWeights | None = None,
        target_causal_density: float = 0.20,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.target_causal_density = target_causal_density
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        causal_mask: torch.Tensor | None = None,
        energy: torch.Tensor | None = None,
        uncertainty: torch.Tensor | None = None,
        plasticity_penalty: torch.Tensor | None = None,
        causal_reg_loss: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, LossBreakdown]:
        task = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)

        causal = torch.zeros((), device=logits.device, dtype=logits.dtype)
        if causal_mask is not None:
            density = causal_mask.mean()
            causal = causal + (density - self.target_causal_density).pow(2)
        if causal_reg_loss is not None:
            causal = causal + causal_reg_loss.to(device=logits.device, dtype=logits.dtype)

        energy_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
        if energy is not None:
            energy_loss = energy.to(device=logits.device, dtype=logits.dtype).mean()

        uncertainty_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
        if uncertainty is not None:
            uncertainty_loss = uncertainty.to(device=logits.device, dtype=logits.dtype).mean()

        consolidation = torch.zeros((), device=logits.device, dtype=logits.dtype)
        if plasticity_penalty is not None:
            consolidation = plasticity_penalty.to(device=logits.device, dtype=logits.dtype)

        weights = self.weights
        total = (
            weights.task * task
            + weights.causal * causal
            + weights.energy * energy_loss
            + weights.uncertainty * uncertainty_loss
            + weights.consolidation * consolidation
        )

        breakdown = LossBreakdown(
            total=float(total.detach().cpu()),
            task=float(task.detach().cpu()),
            causal=float(causal.detach().cpu()),
            energy=float(energy_loss.detach().cpu()),
            uncertainty=float(uncertainty_loss.detach().cpu()),
            consolidation=float(consolidation.detach().cpu()),
        )
        return total, breakdown


CSPNLoss = CausperaLoss

