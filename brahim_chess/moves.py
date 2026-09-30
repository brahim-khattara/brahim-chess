"""Square indexing and legal-move masking helpers."""

from __future__ import annotations

import chess
import torch


def square_to_index(square_str: str) -> int:
    file_idx = ord(square_str[0]) - ord("a")
    rank_idx = int(square_str[1]) - 1
    return rank_idx * 8 + file_idx


def index_to_square(square_idx: int) -> str:
    return chess.square_name(square_idx)


def make_uci(start_idx: int, end_idx: int) -> str:
    return f"{index_to_square(start_idx)}{index_to_square(end_idx)}"


def flat_move_index(start_idx: int, end_idx: int) -> int:
    return (start_idx * 64) + end_idx


def split_flat_index(flat_idx: int) -> tuple[int, int]:
    return flat_idx // 64, flat_idx % 64


def legal_move_mask(fen_str: str) -> torch.Tensor:
    """Return a length-4096 float mask with 1.0 on legal (from, to) pairs."""
    board = chess.Board(fen_str)
    mask = torch.zeros(4096, dtype=torch.float32)
    for move in board.legal_moves:
        flat_idx = (move.from_square * 64) + move.to_square
        mask[flat_idx] = 1.0
    return mask
