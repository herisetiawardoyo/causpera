from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch import nn
from torch.nn import functional as F


class AdaptiveSynapticConsolidation:
    """Adaptive parameter-importance tracker for continual learning.

    The tracker supports two modes:
    - Online updates from gradients after `loss.backward()`.
    - Empirical Fisher estimation from a dataloader after a task is complete.

    Both modes maintain an importance map and penalize movement away from the
    latest consolidated parameter reference.
    """

    def __init__(
        self,
        model: nn.Module,
        strength: float = 1e-4,
        decay: float = 0.98,
        eps: float = 1e-8,
    ) -> None:
        self.model = model
        self.strength = strength
        self.decay = decay
        self.eps = eps
        self.reference: dict[str, torch.Tensor] = {}
        self.importance: dict[str, torch.Tensor] = {}
        self.task_count = 0
        self._initialize_buffers()

    def _named_trainable_parameters(self) -> Iterable[tuple[str, nn.Parameter]]:
        return (
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        )

    def _initialize_buffers(self) -> None:
        self.reference.clear()
        self.importance.clear()
        for name, parameter in self._named_trainable_parameters():
            self.reference[name] = parameter.detach().clone().cpu()
            self.importance[name] = torch.zeros_like(parameter.detach(), device="cpu")

    @staticmethod
    def _causal_scale(causal_strength: torch.Tensor | float | None) -> float:
        if causal_strength is None:
            return 1.0
        if isinstance(causal_strength, torch.Tensor):
            return 1.0 + float(causal_strength.detach().mean().clamp(0.0, 1.0).cpu())
        return 1.0 + max(0.0, min(1.0, float(causal_strength)))

    @torch.no_grad()
    def update_importance(self, causal_strength: torch.Tensor | float | None = None) -> None:
        """Update importance from currently stored gradients."""

        scale = self._causal_scale(causal_strength)
        for name, parameter in self._named_trainable_parameters():
            if parameter.grad is None:
                continue
            if name not in self.importance:
                self.reference[name] = parameter.detach().clone().cpu()
                self.importance[name] = torch.zeros_like(parameter.detach(), device="cpu")

            grad_signal = parameter.grad.detach().pow(2).to(self.importance[name].device) * scale
            self.importance[name].mul_(self.decay).add_(grad_signal, alpha=1.0 - self.decay)

    @torch.no_grad()
    def consolidate(self) -> None:
        """Snapshot current trainable parameters as the new reference."""

        for name, parameter in self._named_trainable_parameters():
            self.reference[name] = parameter.detach().clone().cpu()
            self.importance.setdefault(name, torch.zeros_like(parameter.detach(), device="cpu"))
        self.task_count += 1

    def penalty(self) -> torch.Tensor:
        """Return the quadratic consolidation penalty."""

        first_parameter = next(self.model.parameters())
        total = torch.zeros((), device=first_parameter.device)

        for name, parameter in self._named_trainable_parameters():
            if name not in self.reference or name not in self.importance:
                continue
            reference = self.reference[name].to(parameter.device)
            importance = self.importance[name].to(parameter.device)
            total = total + (importance * (parameter - reference).pow(2)).sum()

        return self.strength * total

    def estimate_fisher(
        self,
        dataloader: Iterable,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        max_batches: int | None = None,
        device: torch.device | str | None = None,
        accumulate: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Estimate an empirical Fisher diagonal from a dataloader."""

        if device is None:
            device = next(self.model.parameters()).device
        device = torch.device(device)
        loss_fn = loss_fn or F.cross_entropy

        fisher = {
            name: torch.zeros_like(parameter.detach(), device=device)
            for name, parameter in self._named_trainable_parameters()
        }
        seen = 0
        was_training = self.model.training
        self.model.eval()

        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            if isinstance(batch, dict):
                inputs, targets = batch["input"], batch["target"]
            else:
                inputs, targets = batch[0], batch[1]

            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size = inputs.size(0)

            self.model.zero_grad(set_to_none=True)
            output = self.model(inputs)
            logits = output.logits if hasattr(output, "logits") else output[0]
            loss = loss_fn(logits, targets)
            loss.backward()

            for name, parameter in self._named_trainable_parameters():
                if parameter.grad is not None:
                    fisher[name] += parameter.grad.detach().pow(2) * batch_size
            seen += batch_size

        if was_training:
            self.model.train()

        if seen > 0:
            for name in fisher:
                fisher[name] = fisher[name] / seen

        if accumulate:
            with torch.no_grad():
                for name, value in fisher.items():
                    current = self.importance.get(name)
                    if current is None:
                        self.importance[name] = value.detach().cpu()
                    else:
                        current.mul_(self.decay).add_(value.detach().to(current.device), alpha=1.0 - self.decay)
                self.consolidate()

        return {name: value.detach().cpu() for name, value in fisher.items()}

    def state_dict(self) -> dict[str, object]:
        return {
            "strength": self.strength,
            "decay": self.decay,
            "task_count": self.task_count,
            "reference": {name: value.clone() for name, value in self.reference.items()},
            "importance": {name: value.clone() for name, value in self.importance.items()},
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.strength = float(state.get("strength", self.strength))
        self.decay = float(state.get("decay", self.decay))
        self.task_count = int(state.get("task_count", 0))
        self.reference = {
            str(name): value.clone()
            for name, value in dict(state.get("reference", {})).items()
            if isinstance(value, torch.Tensor)
        }
        self.importance = {
            str(name): value.clone()
            for name, value in dict(state.get("importance", {})).items()
            if isinstance(value, torch.Tensor)
        }


OnlineSynapticConsolidation = AdaptiveSynapticConsolidation
