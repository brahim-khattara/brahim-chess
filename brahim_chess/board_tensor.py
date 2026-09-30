"""Spatial board tensor encoding for the CNN experiment."""

from __future__ import annotations

import chess
import torch

PIECE_CHANNELS = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def get_last_move_uci(row) -> str:
    for col in [
        "Opponent_Last_Move_UCI",
        "Last_Opponent_Move_UCI",
        "Last_Move_UCI",
        "Prev_Move_UCI",
    ]:
        if col in row and isinstance(row[col], str) and row[col].strip():
            return row[col]
    return "none"


def board_to_tensor(fen_str: str, last_move_uci: str = "none") -> torch.Tensor:
    """Encode a position as 14×8×8: 12 piece planes + 2 last-move planes."""
    board = chess.Board(fen_str)
    tensor = torch.zeros(14, 8, 8, dtype=torch.float32)

    for square, piece in board.piece_map().items():
        base = PIECE_CHANNELS[piece.piece_type]
        channel = base if piece.color == chess.WHITE else base + 6
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        tensor[channel, rank, file] = 1.0

    if isinstance(last_move_uci, str) and len(last_move_uci) >= 4:
        try:
            move = chess.Move.from_uci(last_move_uci[:4])
            tensor[12, chess.square_rank(move.from_square), chess.square_file(move.from_square)] = 1.0
            tensor[13, chess.square_rank(move.to_square), chess.square_file(move.to_square)] = 1.0
        except ValueError:
            pass

    return tensor
