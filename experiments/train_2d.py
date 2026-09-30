"""Train / fine-tune the production BrahimClone2D architecture.

Used by EXP-12 (train from scratch), EXP-14 (history + Elo≥1400), and
EXP-15 (fine-tune EXP-12 weights on Elo≥1400 data).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from brahim_chess.dataset import UnpackedFenDataset
from brahim_chess.inference import get_device
from brahim_chess.models import BrahimClone2D
from brahim_chess.paths import repo_path
from brahim_chess.training import run_dual_head_epoch
from brahim_chess.viz import collect_dual_head_examples, plot_loss_and_exact, save_examples


def train_brahim_clone_2d(
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    *,
    exp_name: str,
    use_history: bool = True,
    init_weights: Path | None = None,
    epochs: int = 150,
    batch_size: int = 128,
    lr: float = 1e-4,
    patience: int = 8,
    save_dir: Path | None = None,
    plot_dir: Path | None = None,
) -> None:
    device = get_device()
    print(f"[{exp_name}] Training on device: {device}")

    save_dir = save_dir or repo_path("models")
    plot_dir = plot_dir or repo_path("plots")
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        UnpackedFenDataset(str(train_csv), use_history=use_history),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        UnpackedFenDataset(str(val_csv), use_history=use_history),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        UnpackedFenDataset(str(test_csv), use_history=use_history),
        batch_size=batch_size,
        shuffle=False,
    )

    model = BrahimClone2D().to(device)
    if init_weights is not None:
        print(f"Loading initial weights from {init_weights}")
        state = torch.load(init_weights, map_location=device, weights_only=True)
        model.load_state_dict(state)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_path = save_dir / f"brahim_clone_{exp_name}_best.pth"
    last_path = save_dir / f"brahim_clone_{exp_name}_last.pth"

    history = {"train_loss": [], "val_loss": [], "train_exact": [], "val_exact": []}
    best_val_exact = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss, _, _, train_exact, _ = run_dual_head_epoch(
            model, train_loader, optimizer, device, masked_eval=False
        )
        val_loss, _, _, val_exact, _ = run_dual_head_epoch(
            model, val_loader, None, device, masked_eval=False
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_exact"].append(train_exact)
        history["val_exact"].append(val_exact)

        print(
            f"Epoch {epoch:02d} | train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | val_exact {val_exact:.2f}%"
        )
        scheduler.step(val_loss)

        if val_exact > best_val_exact:
            best_val_exact = val_exact
            torch.save(model.state_dict(), best_path)
            epochs_without_improvement = 0
            print("  -> New best model saved!")
        else:
            epochs_without_improvement += 1
            print(f"  -> No improvement for {epochs_without_improvement} epoch(s).")
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break

    torch.save(model.state_dict(), last_path)
    plot_loss_and_exact(history, str(plot_dir), prefix=exp_name, title=exp_name.upper())

    print("\nTesting best model (masked evaluation)...")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_loss, _, _, test_exact, test_exact_masked = run_dual_head_epoch(
        model, test_loader, None, device, masked_eval=True
    )
    print(
        f"FINAL RESULTS: Test loss {test_loss:.4f} | "
        f"exact {test_exact:.2f}% | exact_masked {test_exact_masked:.2f}%"
    )

    correct, wrong = collect_dual_head_examples(model, test_loader, device)
    save_examples(correct, wrong, str(plot_dir / f"{exp_name}_qualitative"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", default="exp15_finetuned")
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=repo_path("data", "splits", "brahim_with_history_10min_elo1400_training_set.csv"),
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=repo_path("data", "splits", "brahim_with_history_10min_elo1400_validation_set.csv"),
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=repo_path("data", "splits", "brahim_with_history_10min_elo1400_test_set.csv"),
    )
    parser.add_argument(
        "--init-weights",
        type=Path,
        default=repo_path("models", "brahim_clone_exp12_best.pth"),
    )
    parser.add_argument("--no-init", action="store_true", help="Train from scratch")
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    train_brahim_clone_2d(
        args.train_csv,
        args.val_csv,
        args.test_csv,
        exp_name=args.exp_name,
        use_history=not args.no_history,
        init_weights=None if args.no_init else args.init_weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
