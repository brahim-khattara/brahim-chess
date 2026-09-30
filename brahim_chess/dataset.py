"""PyTorch datasets for behavior-cloning experiments."""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset

from brahim_chess.board_tensor import board_to_tensor, get_last_move_uci
from brahim_chess.fen import tokenize_raw_fen, tokenize_unpacked_fen
from brahim_chess.moves import flat_move_index, square_to_index


class RawFenDataset(Dataset):
    """Character-level raw FEN → start/end square targets (EXP-01..04)."""

    def __init__(self, csv_filepath: str):
        print(f"Loading data from {csv_filepath}...")
        self.data = pd.read_csv(csv_filepath)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]
        fen_str = row["Board_State_FEN"]
        uci_move = row["Brahim_Move_UCI"]
        tokens = tokenize_raw_fen(fen_str)
        start_sq = square_to_index(uci_move[:2])
        end_sq = square_to_index(uci_move[2:4])
        return {
            "input_ids": tokens,
            "start_target": torch.tensor(start_sq, dtype=torch.long),
            "end_target": torch.tensor(end_sq, dtype=torch.long),
            "fen_str": fen_str,
            "uci_move": uci_move,
        }


class UnifiedActionDataset(Dataset):
    """Raw FEN → flat 4096-way move target (EXP-07)."""

    def __init__(self, csv_filepath: str):
        print(f"Loading data from {csv_filepath}...")
        self.data = pd.read_csv(csv_filepath)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]
        fen_str = row["Board_State_FEN"]
        uci_move = row["Brahim_Move_UCI"]
        tokens = tokenize_raw_fen(fen_str)
        start_sq = square_to_index(uci_move[:2])
        end_sq = square_to_index(uci_move[2:4])
        return {
            "input_ids": tokens,
            "move_target": torch.tensor(flat_move_index(start_sq, end_sq), dtype=torch.long),
            "fen_str": fen_str,
            "uci_move": uci_move,
        }


class UnpackedFenDataset(Dataset):
    """Unpacked board + optional opponent history (EXP-12/14/15)."""

    def __init__(self, csv_filepath: str, use_history: bool = True):
        print(f"Loading data from {csv_filepath}...")
        self.data = pd.read_csv(csv_filepath)
        self.use_history = use_history

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]
        fen_str = row["Board_State_FEN"]
        uci_move = row["Brahim_Move_UCI"]
        last_move = ""
        if self.use_history:
            last_move = row.get("Opponent_Last_Move_UCI", "")
            if pd.isna(last_move):
                last_move = ""
        tokens = tokenize_unpacked_fen(fen_str, last_move if last_move else "none")
        start_sq = square_to_index(uci_move[:2])
        end_sq = square_to_index(uci_move[2:4])
        return {
            "input_ids": tokens,
            "start_target": torch.tensor(start_sq, dtype=torch.long),
            "end_target": torch.tensor(end_sq, dtype=torch.long),
            "fen_str": fen_str,
            "uci_move": uci_move,
        }


class BoardTensorDataset(Dataset):
    """14×8×8 spatial tensor dataset (EXP-13 CNN)."""

    def __init__(self, csv_filepath: str):
        print(f"Loading data from {csv_filepath}...")
        self.data = pd.read_csv(csv_filepath)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]
        fen_str = row["Board_State_FEN"]
        uci_move = row["Brahim_Move_UCI"]
        last_move = get_last_move_uci(row)
        start_sq = square_to_index(uci_move[:2])
        end_sq = square_to_index(uci_move[2:4])
        return {
            "input_tensor": board_to_tensor(fen_str, last_move_uci=last_move),
            "start_target": torch.tensor(start_sq, dtype=torch.long),
            "end_target": torch.tensor(end_sq, dtype=torch.long),
            "fen_str": fen_str,
            "uci_move": uci_move,
        }


# Back-compat alias used by older experiment scripts.
ChessBehaviorDataset = UnpackedFenDataset

__all__ = [
    "RawFenDataset",
    "UnifiedActionDataset",
    "UnpackedFenDataset",
    "BoardTensorDataset",
    "ChessBehaviorDataset",
]
