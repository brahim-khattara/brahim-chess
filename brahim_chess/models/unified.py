"""Unified 4096-way action head (EXP-07)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from brahim_chess.models.baseline import PositionalEncoding
from brahim_chess.vocab import PAD_IDX, RAW_VOCAB


class BrahimCloneUnified(nn.Module):
    def __init__(
        self,
        vocab_size: int = len(RAW_VOCAB) + 1,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.move_head = nn.Linear(d_model, 4096)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        memory = self.transformer_encoder(x)
        board_rep = memory.mean(dim=1)
        return self.move_head(board_rep)
