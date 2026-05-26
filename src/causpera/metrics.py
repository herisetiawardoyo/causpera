from __future__ import annotations

import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == targets).float().mean().detach().cpu())


def count_routes(path_index: torch.Tensor, num_paths: int = 2) -> list[int]:
    counts = torch.bincount(path_index.detach().cpu(), minlength=num_paths)
    return [int(value) for value in counts[:num_paths]]

