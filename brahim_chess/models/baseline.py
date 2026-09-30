"""Sinusoidal positional encoding + baseline uncoupled dual-head Transformer."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from brahim_chess.vocab import PAD_IDX, RAW_VOCAB


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 120):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(0)]


class BrahimClone(nn.Module):
    """EXP-01 / EXP-03 baseline: mean-pooled Transformer → independent start/end heads."""

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
        self.start_head = nn.Linear(d_model, 64)
        self.end_head = nn.Linear(d_model, 64)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        memory = self.transformer_encoder(x)
        board_rep = memory.mean(dim=1)
        return self.start_head(board_rep), self.end_head(board_rep)
