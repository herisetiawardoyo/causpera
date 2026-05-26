from __future__ import annotations

import torch

from causpera import CausperaClassifier, CausperaConfig


def small_config() -> CausperaConfig:
    return CausperaConfig(
        input_dim=8,
        d_model=32,
        d_ff=64,
        num_classes=2,
        num_layers_fast=1,
        num_layers_total=2,
        num_heads=4,
        dropout=0.0,
        mc_samples=2,
        max_seq_len=32,
    )


def test_model_forward_and_loss() -> None:
    torch.manual_seed(1)
    model = CausperaClassifier(small_config())
    x = torch.randn(5, 12, 8)
    y = torch.randint(0, 2, (5,))

    output = model(x)
    loss, parts = model.compute_loss(output, y)
    loss.backward()

    assert output.logits.shape == (5, 2)
    assert output.hidden.shape == (5, 12, 32)
    assert output.causal_mask.shape == (5, 4, 12, 12)
    assert output.attention.shape == (5, 4, 12, 12)
    assert output.causal_reg_loss.ndim == 0
    assert parts.task > 0
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_model_infer_routes_batch() -> None:
    model = CausperaClassifier(small_config())
    x = torch.randn(6, 10, 8)

    output = model.infer(x)

    assert output.logits.shape == (6, 2)
    assert output.uncertainty.shape == (6,)
    assert output.path_index.shape == (6,)
    assert output.stats.total == 6
    assert 0.0 <= output.stats.fast_ratio <= 1.0
    assert 0.0 <= output.stats.estimated_compute_savings <= 1.0


def test_get_causal_graph_shape() -> None:
    model = CausperaClassifier(small_config())
    x = torch.randn(1, 7, 8)

    graph = model.get_causal_graph(x, layer_idx=0, head_idx=0)

    assert graph.shape == (7, 7)


def test_compatibility_aliases() -> None:
    from causpera import CSPN, CSPNConfig

    model = CSPN(CSPNConfig(input_dim=8, d_model=32, d_ff=64, num_classes=2, num_heads=4))

    assert isinstance(model, CausperaClassifier)

