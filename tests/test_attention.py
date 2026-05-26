from __future__ import annotations

import torch

from causpera.attention import CausalMaskEstimator, CausalSparseAttention


def test_causal_attention_shapes_and_gradients() -> None:
    attention = CausalSparseAttention(d_model=32, num_heads=4, dropout=0.0, max_lag=3)
    x = torch.randn(2, 6, 32, requires_grad=True)

    out, info = attention(x)
    loss = out.sum() + info.reg_loss
    loss.backward()

    assert out.shape == (2, 6, 32)
    assert info.attention.shape == (2, 4, 6, 6)
    assert info.causal_mask.shape == (2, 4, 6, 6)
    assert info.reg_loss.ndim == 0
    assert x.grad is not None
    assert torch.isfinite(out).all()


def test_temporal_prior_blocks_future_sources() -> None:
    estimator = CausalMaskEstimator(d_model=16, num_heads=2, max_lag=2)
    x = torch.randn(1, 5, 16)
    mask = estimator(x)

    future_positions = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    assert torch.all(mask[0, 0][future_positions] == 0)


def test_attention_does_not_attend_to_future_sources() -> None:
    attention = CausalSparseAttention(d_model=16, num_heads=2, dropout=0.0, max_lag=2)
    x = torch.randn(1, 5, 16)
    _, info = attention(x)

    future_positions = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    assert torch.all(info.attention[0, 0][future_positions] == 0)


def test_padding_mask_keeps_output_finite() -> None:
    attention = CausalSparseAttention(d_model=16, num_heads=2, dropout=0.0)
    x = torch.randn(2, 4, 16)
    mask = torch.tensor([[True, True, False, False], [False, False, False, False]])

    out, info = attention(x, attention_mask=mask)

    assert out.shape == (2, 4, 16)
    assert torch.isfinite(out).all()
    assert torch.isfinite(info.attention).all()
