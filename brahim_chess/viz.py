"""Plotting and qualitative board SVG helpers."""

from __future__ import annotations

import os
from pathlib import Path

import chess
import chess.svg
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from brahim_chess.moves import legal_move_mask, make_uci


def plot_loss_and_exact(history: dict, plot_dir: str, prefix: str, title: str) -> None:
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} — Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{prefix}_loss_curves.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_exact"], label="Train Exact (Unmasked)")
    plt.plot(epochs, history["val_exact"], label="Val Exact (Unmasked)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{title} — Exact Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{prefix}_exact_accuracy.png"), dpi=300)
    plt.close()


def render_board_svg(fen_str: str, predicted_uci: str, actual_uci: str | None, out_path: str) -> None:
    board = chess.Board(fen_str)
    arrows = []
    try:
        pred_move = chess.Move.from_uci(predicted_uci)
        arrows.append(chess.svg.Arrow(pred_move.from_square, pred_move.to_square, color="#d62728"))
    except ValueError:
        pass
    if actual_uci is not None:
        try:
            actual_move = chess.Move.from_uci(actual_uci)
            arrows.append(chess.svg.Arrow(actual_move.from_square, actual_move.to_square, color="#2ca02c"))
        except ValueError:
            pass
    svg = chess.svg.board(board=board, arrows=arrows, size=400, coordinates=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(svg)


def collect_dual_head_examples(model, dataloader, device, max_correct=6, max_wrong=6, input_key="input_ids"):
    model.eval()
    correct, wrong = [], []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[input_key].to(device)
            start_targets = batch["start_target"].to(device)
            end_targets = batch["end_target"].to(device)
            start_logits, end_logits = model(inputs)
            start_probs = F.softmax(start_logits, dim=1)
            end_probs = F.softmax(end_logits, dim=1)

            for i in range(len(batch["fen_str"])):
                joint = torch.outer(start_probs[i], end_probs[i]).reshape(-1)
                mask = legal_move_mask(batch["fen_str"][i]).to(joint.device)
                joint = joint * mask
                pred_flat = int(joint.argmax().item())
                pred_uci = make_uci(pred_flat // 64, pred_flat % 64)
                actual_short = batch["uci_move"][i][:4]
                if pred_uci == actual_short and len(correct) < max_correct:
                    correct.append((batch["fen_str"][i], pred_uci, actual_short))
                elif pred_uci != actual_short and len(wrong) < max_wrong:
                    wrong.append((batch["fen_str"][i], pred_uci, actual_short))
                if len(correct) >= max_correct and len(wrong) >= max_wrong:
                    return correct, wrong
    return correct, wrong


def save_examples(correct, wrong, out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for i, (fen, pred, actual) in enumerate(correct, start=1):
        render_board_svg(fen, pred, actual, os.path.join(out_dir, f"correct_{i:02d}.svg"))
    for i, (fen, pred, actual) in enumerate(wrong, start=1):
        render_board_svg(fen, pred, actual, os.path.join(out_dir, f"wrong_{i:02d}.svg"))
