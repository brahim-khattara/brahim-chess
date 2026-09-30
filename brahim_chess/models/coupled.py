"""Coupled start→end heads (EXP-04 / EXP-08)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from brahim_chess.models.baseline import PositionalEncoding
from brahim_chess.vocab import PAD_IDX, RAW_VOCAB


class BrahimCloneCoupled(nn.Module):
    def __init__(
        self,
        vocab_size: int = len(RAW_VOCAB) + 1,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        start_embed_dim: int = 32,
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
        self.start_embed = nn.Embedding(64, start_embed_dim)
        self.end_head = nn.Linear(d_model + start_embed_dim, 64)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        memory = self.transformer_encoder(x)
        board_rep = memory.mean(dim=1)
        start_logits = self.start_head(board_rep)
        start_probs = F.softmax(start_logits, dim=1)
        start_emb = start_probs @ self.start_embed.weight
        end_input = torch.cat([board_rep, start_emb], dim=1)
        end_logits = self.end_head(end_input)
        return start_logits, end_logits
