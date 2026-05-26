from __future__ import annotations

import torch

from causpera.losses import CausperaLoss, LossWeights


def test_loss_is_differentiable() -> None:
    criterion = CausperaLoss(weights=LossWeights(causal=0.1))
    logits = torch.randn(4, 3, requires_grad=True)
    targets = torch.randint(0, 3, (4,))
    causal_mask = torch.rand(4, 2, 5, 5)

    loss, parts = criterion(logits=logits, targets=targets, causal_mask=causal_mask)
    loss.backward()

    assert loss.ndim == 0
    assert parts.total > 0
    assert parts.task > 0
    assert logits.grad is not None


def test_loss_accepts_optional_terms() -> None:
    criterion = CausperaLoss()
    logits = torch.randn(4, 2)
    targets = torch.randint(0, 2, (4,))
    energy = torch.rand(4)
    uncertainty = torch.rand(4)
    plasticity = torch.tensor(0.5)

    loss, parts = criterion(
        logits=logits,
        targets=targets,
        energy=energy,
        uncertainty=uncertainty,
        plasticity_penalty=plasticity,
    )

    assert loss.item() > 0
    assert parts.energy >= 0
    assert parts.uncertainty >= 0
    assert parts.consolidation == 0.5

