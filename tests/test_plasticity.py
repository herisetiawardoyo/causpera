from __future__ import annotations

import torch

from causpera import AdaptiveSynapticConsolidation, CausperaClassifier, CausperaConfig


def test_plasticity_penalty_changes_after_parameter_shift() -> None:
    model = CausperaClassifier(
        CausperaConfig(
            input_dim=4,
            d_model=16,
            d_ff=32,
            num_classes=2,
            num_layers_fast=1,
            num_layers_total=1,
            num_heads=4,
        )
    )
    tracker = AdaptiveSynapticConsolidation(model, strength=1.0, decay=0.0)
    x = torch.randn(4, 5, 4)
    y = torch.randint(0, 2, (4,))

    output = model(x)
    loss, _ = model.compute_loss(output, y, plasticity_penalty=torch.zeros(()))
    loss.backward()
    tracker.update_importance(causal_strength=output.causal_mask.mean())
    tracker.consolidate()

    assert tracker.penalty().item() == 0.0

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))

    assert tracker.penalty().item() > 0.0


def test_fisher_estimation_runs() -> None:
    model = CausperaClassifier(
        CausperaConfig(
            input_dim=4,
            d_model=16,
            d_ff=32,
            num_classes=2,
            num_layers_fast=1,
            num_layers_total=1,
            num_heads=4,
        )
    )
    tracker = AdaptiveSynapticConsolidation(model, strength=1.0)
    x = torch.randn(8, 5, 4)
    y = torch.randint(0, 2, (8,))
    loader = [(x, y)]

    fisher = tracker.estimate_fisher(loader, max_batches=1)

    assert fisher
    assert tracker.task_count == 1

