from __future__ import annotations

from causpera.data import ToyTemporalCausalDataset, make_toy_loaders


def test_toy_dataset_shapes() -> None:
    dataset = ToyTemporalCausalDataset(n_samples=32, seq_len=8, input_dim=6, seed=1)
    x, y = dataset[0]

    assert len(dataset) == 32
    assert x.shape == (8, 6)
    assert y.ndim == 0
    assert len(dataset.true_edges) == 3


def test_make_toy_loaders() -> None:
    train_loader, val_loader, dataset = make_toy_loaders(n_samples=64, batch_size=8, seed=1)

    assert len(dataset) == 64
    assert len(train_loader) > 0
    assert len(val_loader) > 0

