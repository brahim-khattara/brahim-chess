"""Masked Top-1 / Top-5 evaluation for BrahimClone2D checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from brahim_chess.dataset import UnpackedFenDataset
from brahim_chess.inference import get_device, load_brahim_clone_2d
from brahim_chess.metrics import topk_masked_dual
from brahim_chess.paths import repo_path


@torch.no_grad()
def eval_topk_masked(model, dataloader, device, k: int = 5) -> tuple[float, float]:
    model.eval()
    total = 0
    top1 = 0
    topk_hits = 0

    print("Evaluating masked Top-K...")
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        start_logits, end_logits = model(input_ids)
        for i in range(input_ids.size(0)):
            actual = batch["uci_move"][i][:4]
            ranked = topk_masked_dual(
                start_logits[i].cpu(),
                end_logits[i].cpu(),
                batch["fen_str"][i],
                k=k,
            )
            preds = [uci for uci, _ in ranked]
            if preds and preds[0] == actual:
                top1 += 1
            if actual in preds:
                topk_hits += 1
            total += 1

    return 100.0 * top1 / max(total, 1), 100.0 * topk_hits / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=repo_path("models", "brahim_clone_exp15_finetuned_best.pth"),
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=repo_path("data", "splits", "brahim_with_history_10min_elo1400_test_set.csv"),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--no-history", action="store_true")
    args = parser.parse_args()

    device = get_device()
    model = load_brahim_clone_2d(args.weights, device)
    loader = DataLoader(
        UnpackedFenDataset(str(args.test_csv), use_history=not args.no_history),
        batch_size=args.batch_size,
        shuffle=False,
    )

    top1_acc, topk_acc = eval_topk_masked(model, loader, device, k=args.k)
    print(f"\n--- Masked Evaluation ({args.weights.name}) ---")
    print(f"Top-1 Masked Exact Accuracy: {top1_acc:.2f}%")
    print(f"Top-{args.k} Masked Exact Accuracy: {topk_acc:.2f}%")
    print(f"Delta (Top-{args.k} - Top-1): {topk_acc - top1_acc:.2f}%")


if __name__ == "__main__":
    main()
