from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from causpera.attention import CausalAttentionInfo, CausalSparseAttention
from causpera.config import CausperaConfig
from causpera.losses import CausperaLoss, LossWeights
from causpera.plasticity import AdaptiveSynapticConsolidation


@dataclass
class RouteStats:
    fast_count: int
    slow_count: int
    mean_uncertainty: float
    average_relative_cost: float

    @property
    def total(self) -> int:
        return self.fast_count + self.slow_count

    @property
    def fast_ratio(self) -> float:
        return self.fast_count / max(self.total, 1)

    @property
    def estimated_compute_savings(self) -> float:
        return max(0.0, 1.0 - self.average_relative_cost)


@dataclass
class CausperaOutput:
    logits: torch.Tensor
    hidden: torch.Tensor
    pooled: torch.Tensor
    causal_mask: torch.Tensor
    attention: torch.Tensor
    causal_reg_loss: torch.Tensor
    sparsity: torch.Tensor


@dataclass
class InferenceOutput:
    logits: torch.Tensor
    uncertainty: torch.Tensor
    path_index: torch.Tensor
    stats: RouteStats


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CausperaEncoderBlock(nn.Module):
    def __init__(self, config: CausperaConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attention = CausalSparseAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            dropout=config.dropout,
            max_lag=config.max_lag,
            causal_temperature=config.causal_temperature,
            causal_threshold=config.causal_threshold,
            causal_floor=config.causal_floor,
            enforce_temporal_order=config.enforce_temporal_order,
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.d_ff, config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, CausalAttentionInfo]:
        attention_out, info = self.attention(self.norm1(x), attention_mask=attention_mask)
        x = x + attention_out
        x = x + self.ff(self.norm2(x))
        if attention_mask is not None:
            x = x * attention_mask[:, :, None].to(dtype=x.dtype)
        return x, info


class CausperaClassifier(nn.Module):
    """Sequence classifier integrating causal attention, plasticity, and routing."""

    def __init__(self, config: CausperaConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Sequential(
            nn.Linear(config.input_dim, config.d_model),
            nn.LayerNorm(config.d_model),
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_model))
        self.layers = nn.ModuleList([CausperaEncoderBlock(config) for _ in range(config.num_layers_total)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, config.num_classes)
        self.loss_fn = CausperaLoss(
            weights=LossWeights(
                task=config.loss_task_weight,
                causal=config.loss_causal_weight,
                energy=config.loss_energy_weight,
                uncertainty=config.loss_uncertainty_weight,
                consolidation=config.loss_consolidation_weight,
            ),
            target_causal_density=config.target_causal_density,
            label_smoothing=config.label_smoothing,
        )
        self._init_weights()
        self.plasticity = AdaptiveSynapticConsolidation(
            self,
            strength=config.plasticity_strength,
            decay=config.plasticity_decay,
        )

    def _init_weights(self) -> None:
        nn.init.normal_(self.pos_emb, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("x must have shape [batch, seq_len, input_dim].")
        if x.size(-1) != self.config.input_dim:
            raise ValueError(f"expected input_dim={self.config.input_dim}, got {x.size(-1)}.")
        if x.size(1) > self.config.max_seq_len:
            raise ValueError(f"seq_len={x.size(1)} exceeds max_seq_len={self.config.max_seq_len}.")
        return self.input_proj(x) + self.pos_emb[:, : x.size(1), :]

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.final_norm(hidden)
        if attention_mask is None:
            return hidden.mean(dim=1)
        weights = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _run_layers(
        self,
        hidden: torch.Tensor,
        start: int,
        end: int,
        attention_mask: torch.Tensor | None,
        collect_info: bool,
    ) -> tuple[torch.Tensor, list[CausalAttentionInfo]]:
        infos: list[CausalAttentionInfo] = []
        for layer in self.layers[start:end]:
            hidden, info = layer(hidden, attention_mask=attention_mask)
            if collect_info:
                infos.append(info)
        return hidden, infos

    @staticmethod
    def _normalized_entropy(logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        max_entropy = torch.log(torch.tensor(logits.size(-1), device=logits.device, dtype=logits.dtype))
        return entropy / max_entropy.clamp_min(1e-8)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> CausperaOutput:
        hidden = self._embed(x)
        hidden, infos = self._run_layers(
            hidden=hidden,
            start=0,
            end=self.config.num_layers_total,
            attention_mask=attention_mask,
            collect_info=True,
        )
        pooled = self._pool(hidden, attention_mask)
        logits = self.classifier(pooled)

        causal_mask = torch.stack([info.causal_mask for info in infos], dim=0).mean(dim=0)
        attention = torch.stack([info.attention for info in infos], dim=0).mean(dim=0)
        causal_reg_loss = torch.stack([info.reg_loss for info in infos]).mean()
        sparsity = torch.stack([info.sparsity for info in infos]).mean()

        return CausperaOutput(
            logits=logits,
            hidden=hidden,
            pooled=pooled,
            causal_mask=causal_mask,
            attention=attention,
            causal_reg_loss=causal_reg_loss,
            sparsity=sparsity,
        )

    @torch.no_grad()
    def estimate_uncertainty(
        self,
        fast_hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        samples = []
        for _ in range(self.config.mc_samples):
            noisy = F.dropout(fast_hidden, p=self.config.mc_dropout, training=self.config.mc_samples > 1)
            pooled = self._pool(noisy, attention_mask)
            samples.append(F.softmax(self.classifier(pooled), dim=-1))
        mean_probs = torch.stack(samples, dim=0).mean(dim=0).clamp_min(1e-8)
        entropy = -(mean_probs * mean_probs.log()).sum(dim=-1)
        max_entropy = torch.log(
            torch.tensor(mean_probs.size(-1), device=mean_probs.device, dtype=mean_probs.dtype)
        )
        uncertainty = entropy / max_entropy.clamp_min(1e-8)
        return uncertainty

    @torch.no_grad()
    def infer(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> InferenceOutput:
        was_training = self.training
        self.eval()
        try:
            hidden = self._embed(x)
            fast_hidden, _ = self._run_layers(
                hidden=hidden,
                start=0,
                end=self.config.num_layers_fast,
                attention_mask=attention_mask,
                collect_info=False,
            )
            fast_logits = self.classifier(self._pool(fast_hidden, attention_mask))
            uncertainty = self.estimate_uncertainty(fast_hidden, attention_mask)
            use_fast = uncertainty <= self.config.uncertainty_threshold
            slow_indices = (~use_fast).nonzero(as_tuple=True)[0]
            path_index = torch.where(use_fast, torch.zeros_like(use_fast, dtype=torch.long), torch.ones_like(use_fast, dtype=torch.long))

            final_logits = fast_logits.clone()
            if slow_indices.numel() > 0 and self.config.num_extra_slow_layers > 0:
                slow_hidden = fast_hidden.index_select(0, slow_indices)
                slow_mask = attention_mask.index_select(0, slow_indices) if attention_mask is not None else None
                slow_hidden, _ = self._run_layers(
                    hidden=slow_hidden,
                    start=self.config.num_layers_fast,
                    end=self.config.num_layers_total,
                    attention_mask=slow_mask,
                    collect_info=False,
                )
                slow_logits = self.classifier(self._pool(slow_hidden, slow_mask))
                final_logits.index_copy_(0, slow_indices, slow_logits)

            fast_count = int(use_fast.sum().item())
            slow_count = int((~use_fast).sum().item())
            average_cost = (
                fast_count * self.config.fast_relative_cost + slow_count * 1.0
            ) / max(x.size(0), 1)
            stats = RouteStats(
                fast_count=fast_count,
                slow_count=slow_count,
                mean_uncertainty=float(uncertainty.mean().cpu()),
                average_relative_cost=float(average_cost),
            )
            return InferenceOutput(
                logits=final_logits,
                uncertainty=uncertainty,
                path_index=path_index,
                stats=stats,
            )
        finally:
            if was_training:
                self.train()

    def compute_loss(
        self,
        output: CausperaOutput,
        targets: torch.Tensor,
        plasticity_penalty: torch.Tensor | None = None,
    ):
        if plasticity_penalty is None:
            plasticity_penalty = self.plasticity.penalty()
        return self.loss_fn(
            logits=output.logits,
            targets=targets,
            causal_mask=output.causal_mask,
            plasticity_penalty=plasticity_penalty,
            causal_reg_loss=output.causal_reg_loss,
        )

    def update_importance(self, causal_strength: torch.Tensor | float | None = None) -> None:
        self.plasticity.update_importance(causal_strength=causal_strength)

    def consolidate(self) -> None:
        self.plasticity.consolidate()

    @torch.no_grad()
    def get_causal_graph(
        self,
        x: torch.Tensor,
        layer_idx: int = 0,
        head_idx: int = 0,
    ) -> torch.Tensor:
        if not 0 <= layer_idx < self.config.num_layers_total:
            raise ValueError("layer_idx out of range.")
        if not 0 <= head_idx < self.config.num_heads:
            raise ValueError("head_idx out of range.")

        was_training = self.training
        self.eval()
        try:
            hidden = self._embed(x)
            for index, layer in enumerate(self.layers):
                hidden, info = layer(hidden)
                if index == layer_idx:
                    return info.causal_mask[0, head_idx].detach().cpu()
        finally:
            if was_training:
                self.train()
        raise RuntimeError("unreachable layer traversal state.")

    def num_parameters(self, trainable_only: bool = True) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (parameter for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)


CSPN = CausperaClassifier
CSPNClassifier = CausperaClassifier
CSPNOutput = CausperaOutput
