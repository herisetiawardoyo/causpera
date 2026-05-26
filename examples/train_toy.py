from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from causpera import (
    AdaptiveSynapticConsolidation,
    CausperaClassifier,
    CausperaConfig,
    CausperaLoss,
    accuracy,
    count_routes,
    make_toy_loaders,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Causpera on a toy temporal task.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers-fast", type=int, default=1)
    parser.add_argument("--layers-total", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def evaluate(model: CausperaClassifier, criterion: CausperaLoss, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    fast = 0
    slow = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            loss, _ = criterion(
                logits=output.logits,
                targets=y,
                causal_mask=output.causal_mask,
                causal_reg_loss=output.causal_reg_loss,
            )
            inference = model.infer(x)
            route_counts = count_routes(inference.path_index, num_paths=2)
            fast += route_counts[0]
            slow += route_counts[1]
            total_loss += float(loss.cpu())
            total_acc += accuracy(output.logits, y)

    batches = max(len(loader), 1)
    total_routes = max(fast + slow, 1)
    return {
        "loss": total_loss / batches,
        "accuracy": total_acc / batches,
        "fast_ratio": fast / total_routes,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_loader, val_loader, dataset = make_toy_loaders(
        n_samples=args.samples,
        seq_len=args.seq_len,
        input_dim=args.input_dim,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    config = CausperaConfig(
        input_dim=args.input_dim,
        d_model=args.d_model,
        num_classes=2,
        num_layers_fast=args.layers_fast,
        num_layers_total=args.layers_total,
        num_heads=args.heads,
        max_lag=args.max_lag,
    )
    model = CausperaClassifier(config).to(device)
    criterion = CausperaLoss(
        weights=model.loss_fn.weights,
        target_causal_density=config.target_causal_density,
        label_smoothing=config.label_smoothing,
    )
    asc = AdaptiveSynapticConsolidation(
        model,
        strength=config.plasticity_strength,
        decay=config.plasticity_decay,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

    print(f"device={device}")
    print(f"parameters={model.num_parameters():,}")
    print(f"true_edges={dataset.true_edges}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for x, y in progress:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            output = model(x)
            plasticity = asc.penalty()
            loss, parts = criterion(
                logits=output.logits,
                targets=y,
                causal_mask=output.causal_mask,
                plasticity_penalty=plasticity,
                causal_reg_loss=output.causal_reg_loss,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            asc.update_importance(causal_strength=output.causal_mask.mean())
            optimizer.step()

            batch_acc = accuracy(output.logits, y)
            train_loss += float(loss.detach().cpu())
            train_acc += batch_acc
            progress.set_postfix(loss=f"{parts.total:.4f}", acc=f"{batch_acc:.3f}")

        metrics = evaluate(model, criterion, val_loader, device)
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss / max(len(train_loader), 1):.4f} "
            f"train_acc={train_acc / max(len(train_loader), 1):.3f} "
            f"val_loss={metrics['loss']:.4f} "
            f"val_acc={metrics['accuracy']:.3f} "
            f"fast_ratio={metrics['fast_ratio']:.2%}"
        )

    asc.consolidate()

    if args.save:
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        checkpoint_path = output_dir / "causpera_toy.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "config": config,
                "plasticity": asc.state_dict(),
            },
            checkpoint_path,
        )
        print(f"saved={checkpoint_path}")


if __name__ == "__main__":
    main()

