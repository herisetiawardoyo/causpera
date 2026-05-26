from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, random_split


@dataclass(frozen=True)
class CausalEdge:
    source_lag: int
    feature: int
    weight: float


class ToyTemporalCausalDataset(Dataset):
    """Synthetic temporal classification data with known causal lagged features."""

    def __init__(
        self,
        n_samples: int = 4096,
        seq_len: int = 16,
        input_dim: int = 8,
        noise_std: float = 0.35,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if seq_len < 4:
            raise ValueError("seq_len must be at least 4.")
        if input_dim < 5:
            raise ValueError("input_dim must be at least 5.")

        generator = torch.Generator().manual_seed(seed)
        x = torch.randn(n_samples, seq_len, input_dim, generator=generator)

        x[:, :, 3] = 0.8 * x[:, :, 0] + 0.2 * torch.randn(n_samples, seq_len, generator=generator)
        x[:, :, 4] = -0.6 * x[:, :, 1] + 0.4 * torch.randn(n_samples, seq_len, generator=generator)

        last = seq_len - 1
        causal_score = (
            x[:, last - 1, 0]
            + 0.7 * x[:, last - 2, 1]
            - 0.5 * x[:, last - 3, 2]
            + noise_std * torch.randn(n_samples, generator=generator)
        )
        y = (causal_score > 0).long()

        self.x = x.float()
        self.y = y
        self.true_edges = [
            CausalEdge(source_lag=1, feature=0, weight=1.0),
            CausalEdge(source_lag=2, feature=1, weight=0.7),
            CausalEdge(source_lag=3, feature=2, weight=-0.5),
        ]

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def make_toy_loaders(
    n_samples: int = 4096,
    seq_len: int = 16,
    input_dim: int = 8,
    batch_size: int = 64,
    seed: int = 42,
    train_fraction: float = 0.8,
) -> tuple[DataLoader, DataLoader, ToyTemporalCausalDataset]:
    dataset = ToyTemporalCausalDataset(
        n_samples=n_samples,
        seq_len=seq_len,
        input_dim=input_dim,
        seed=seed,
    )
    train_size = int(train_fraction * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader, dataset

