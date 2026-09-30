"""Inference helpers for the production BrahimClone2D model."""

from __future__ import annotations

from pathlib import Path

import chess
import torch
import torch.nn.functional as F

from brahim_chess.fen import tokenize_unpacked_fen
from brahim_chess.metrics import topk_masked_dual
from brahim_chess.models import BrahimClone2D
from brahim_chess.moves import square_to_index


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_brahim_clone_2d(weights_path: str | Path, device: torch.device | None = None) -> BrahimClone2D:
    device = device or get_device()
    model = BrahimClone2D().to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_move(
    model: BrahimClone2D,
    board: chess.Board,
    device: torch.device | None = None,
    last_move_uci: str = "none",
) -> tuple[chess.Move, float]:
    """Pick the highest-scoring legal move (queen promotions only)."""
    device = device or next(model.parameters()).device
    input_ids = tokenize_unpacked_fen(board.fen(), last_move_uci).unsqueeze(0).to(device)
    start_logits, end_logits = model(input_ids)
    start_probs = F.softmax(start_logits[0], dim=0).cpu()
    end_probs = F.softmax(end_logits[0], dim=0).cpu()

    scored = []
    for move in board.legal_moves:
        uci = move.uci()
        joint = float(start_probs[square_to_index(uci[:2])] * end_probs[square_to_index(uci[2:4])])
        if len(uci) == 5 and uci[4] != "q":
            joint = 0.0
        scored.append((joint, move))
    scored.sort(key=lambda x: x[0], reverse=True)
    best_prob, best_move = scored[0]
    return best_move, best_prob * 100.0


@torch.no_grad()
def predict_topk(
    model: BrahimClone2D,
    fen_str: str,
    device: torch.device | None = None,
    last_move_uci: str = "none",
    k: int = 5,
) -> list[tuple[str, float]]:
    device = device or next(model.parameters()).device
    input_ids = tokenize_unpacked_fen(fen_str, last_move_uci).unsqueeze(0).to(device)
    start_logits, end_logits = model(input_ids)
    return topk_masked_dual(start_logits[0].cpu(), end_logits[0].cpu(), fen_str, k=k)
