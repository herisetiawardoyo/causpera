from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CausperaConfig:
    """Configuration for the Causpera sequence classifier."""

    input_dim: int = 8
    d_model: int = 64
    d_ff: int = 256
    num_classes: int = 2
    num_layers_fast: int = 1
    num_layers_total: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    max_seq_len: int = 256

    max_lag: int = 3
    causal_temperature: float = 0.7
    causal_threshold: float = 0.15
    causal_floor: float = 1e-4
    enforce_temporal_order: bool = True

    uncertainty_threshold: float = 0.55
    mc_samples: int = 3
    mc_dropout: float = 0.1

    loss_task_weight: float = 1.0
    loss_causal_weight: float = 0.02
    loss_energy_weight: float = 0.02
    loss_uncertainty_weight: float = 0.01
    loss_consolidation_weight: float = 1.0
    target_causal_density: float = 0.20
    label_smoothing: float = 0.0

    plasticity_strength: float = 1e-4
    plasticity_decay: float = 0.98

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive.")
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive.")
        if self.num_classes <= 1:
            raise ValueError("num_classes must be at least 2.")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if self.num_layers_fast <= 0:
            raise ValueError("num_layers_fast must be positive.")
        if self.num_layers_total < self.num_layers_fast:
            raise ValueError("num_layers_total must be >= num_layers_fast.")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")
        if self.max_lag <= 0:
            raise ValueError("max_lag must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if not 0.0 <= self.mc_dropout < 1.0:
            raise ValueError("mc_dropout must be in [0, 1).")
        if self.mc_samples <= 0:
            raise ValueError("mc_samples must be positive.")
        if self.causal_temperature <= 0:
            raise ValueError("causal_temperature must be positive.")
        if not 0.0 <= self.causal_threshold <= 1.0:
            raise ValueError("causal_threshold must be in [0, 1].")
        if self.causal_floor <= 0:
            raise ValueError("causal_floor must be positive.")
        if not 0.0 <= self.target_causal_density <= 1.0:
            raise ValueError("target_causal_density must be in [0, 1].")

    @property
    def num_layers_slow(self) -> int:
        """Compatibility alias for earlier CSPN prototypes."""

        return self.num_layers_total

    @property
    def num_extra_slow_layers(self) -> int:
        return self.num_layers_total - self.num_layers_fast

    @property
    def fast_relative_cost(self) -> float:
        return self.num_layers_fast / self.num_layers_total


CSPNConfig = CausperaConfig

