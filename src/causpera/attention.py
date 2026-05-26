from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class CausalAttentionInfo:
    attention: torch.Tensor
    causal_mask: torch.Tensor
    causal_strength: torch.Tensor
    reg_loss: torch.Tensor
    sparsity: torch.Tensor


class CausalMaskEstimator(nn.Module):
    """Differentiable temporal causal-mask proxy.

    The estimator combines directional token compatibility, perturbation
    sensitivity, and a temporal lag prior. It is a practical inductive bias,
    not a formal causal identification algorithm.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_lag: int = 3,
        temperature: float = 0.7,
        intervention_scale: float = 0.05,
        enforce_temporal_order: bool = True,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.max_lag = max_lag
        self.temperature = temperature
        self.intervention_scale = intervention_scale
        self.enforce_temporal_order = enforce_temporal_order

        self.cause_proj = nn.Linear(d_model, d_model)
        self.effect_proj = nn.Linear(d_model, d_model)
        self.intervention_proj = nn.Linear(d_model, d_model)
        self.edge_mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def _temporal_prior(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        target = torch.arange(seq_len, device=device)
        source = torch.arange(seq_len, device=device)
        lag = target[:, None] - source[None, :]
        valid = (lag >= 0) & (lag <= self.max_lag)
        return valid.to(dtype=dtype).unsqueeze(0).unsqueeze(0)

    def _lag_distance(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        target = torch.arange(seq_len, device=device)
        source = torch.arange(seq_len, device=device)
        lag = (target[:, None] - source[None, :]).abs().clamp(max=self.max_lag)
        return (lag.to(dtype=dtype) / max(float(self.max_lag), 1.0)).unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("x must have shape [batch, seq_len, d_model].")

        batch, seq_len, _ = x.shape
        heads = self.num_heads

        cause = self.cause_proj(x).view(batch, seq_len, heads, self.d_head).transpose(1, 2)
        effect = self.effect_proj(x).view(batch, seq_len, heads, self.d_head).transpose(1, 2)

        directional = torch.einsum("bhtd,bhsd->bhts", effect, cause) / math.sqrt(self.d_head)
        directional = torch.sigmoid(directional)

        perturb = self.intervention_scale * torch.tanh(
            self.intervention_proj(x).view(batch, seq_len, heads, self.d_head).transpose(1, 2)
        )
        intervened_cause = cause + perturb
        intervened = torch.einsum("bhtd,bhsd->bhts", effect, intervened_cause)
        intervened = torch.sigmoid(intervened / math.sqrt(self.d_head))
        intervention_delta = (intervened - directional).abs()

        prior = self._temporal_prior(seq_len, x.device, x.dtype).expand(batch, heads, seq_len, seq_len)
        lag_distance = self._lag_distance(seq_len, x.device, x.dtype).expand_as(prior)

        features = torch.stack([directional, intervention_delta, prior, lag_distance], dim=-1)
        logits = self.edge_mlp(features).squeeze(-1)
        mask = torch.sigmoid(logits / self.temperature)

        if self.enforce_temporal_order:
            mask = mask * prior

        return mask


class CausalSparseAttention(nn.Module):
    """Multi-head attention modulated by a sparse causal mask."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        max_lag: int = 3,
        causal_temperature: float = 0.7,
        causal_threshold: float = 0.15,
        causal_floor: float = 1e-4,
        enforce_temporal_order: bool = True,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.causal_threshold = causal_threshold
        self.causal_floor = causal_floor

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.mask_estimator = CausalMaskEstimator(
            d_model=d_model,
            num_heads=num_heads,
            max_lag=max_lag,
            temperature=causal_temperature,
            enforce_temporal_order=enforce_temporal_order,
        )

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = tensor.shape
        return tensor.view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)

    def _sparsify(self, mask: torch.Tensor) -> torch.Tensor:
        if self.causal_threshold <= 0:
            return mask
        hard = (mask >= self.causal_threshold).to(mask.dtype)
        return hard - mask.detach() + mask

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, CausalAttentionInfo]:
        batch, seq_len, _ = x.shape

        query = self._split_heads(self.q_proj(x))
        key = self._split_heads(self.k_proj(x))
        value = self._split_heads(self.v_proj(x))

        logits = torch.einsum("bhtd,bhsd->bhts", query, key) / math.sqrt(self.d_head)

        causal_mask = self.mask_estimator(x)
        sparse_mask = self._sparsify(causal_mask)
        min_value = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(causal_mask <= 0, min_value)
        logits = logits + torch.log(sparse_mask.clamp_min(self.causal_floor))

        if attention_mask is not None:
            valid_keys = attention_mask[:, None, None, :].bool()
            logits = logits.masked_fill(~valid_keys, min_value)
            no_valid_keys = ~valid_keys.any(dim=-1, keepdim=True)
            logits = torch.where(no_valid_keys.expand_as(logits), torch.zeros_like(logits), logits)

        attention = F.softmax(logits.float(), dim=-1).to(dtype=x.dtype)
        attention = self.dropout(attention)

        hidden = torch.einsum("bhts,bhsd->bhtd", attention, value)
        hidden = hidden.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        hidden = self.out_proj(hidden)

        if attention_mask is not None:
            hidden = hidden * attention_mask[:, :, None].to(dtype=hidden.dtype)

        sparsity = (causal_mask < self.causal_threshold).to(dtype=x.dtype).mean()
        info = CausalAttentionInfo(
            attention=attention,
            causal_mask=causal_mask,
            causal_strength=causal_mask.mean(dim=1),
            reg_loss=causal_mask.mean(),
            sparsity=sparsity,
        )
        return hidden, info


CausalSparseMultiheadAttention = CausalSparseAttention
