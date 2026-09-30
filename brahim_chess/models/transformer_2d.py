"""2D board-aware Transformer — production architecture (EXP-12 / 14 / 15)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from brahim_chess.vocab import PAD_IDX, UNPACKED_MAX_LEN, UNPACKED_VOCAB


class PositionalEncoding2D(nn.Module):
    """File/rank embeddings for the first 64 board tokens; learned PE for metadata."""

    def __init__(self, d_model: int = 512, max_len: int = UNPACKED_MAX_LEN):
        super().__init__()
        self.file_embed = nn.Embedding(8, d_model // 2)
        self.rank_embed = nn.Embedding(8, d_model // 2)
        self.meta_embed = nn.Embedding(max_len, d_model)

        files = torch.zeros(64, dtype=torch.long)
        ranks = torch.zeros(64, dtype=torch.long)
        for i in range(64):
            files[i] = i % 8
            ranks[i] = i // 8
        self.register_buffer("files", files)
        self.register_buffer("ranks", ranks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        emb_x = self.file_embed(self.files)
        emb_y = self.rank_embed(self.ranks)
        board_pe = torch.cat([emb_x, emb_y], dim=1)

        if seq_len <= 64:
            return x + board_pe[:seq_len].unsqueeze(0)

        meta_indices = torch.arange(seq_len - 64, device=x.device) + 64
        meta_pe = self.meta_embed(meta_indices)
        full_pe = torch.cat([board_pe, meta_pe], dim=0)
        return x + full_pe.unsqueeze(0)


class BrahimClone2D(nn.Module):
    """Per-square start/end heads with soft start context injected into the end head."""

    def __init__(
        self,
        vocab_size: int = len(UNPACKED_VOCAB) + 1,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_encoder = PositionalEncoding2D(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.context_norm = nn.LayerNorm(d_model)
        self.start_head = nn.Linear(d_model, 1)
        self.end_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        memory = self.transformer_encoder(x)
        board_tokens = memory[:, :64, :]

        start_logits = self.start_head(board_tokens).squeeze(-1)
        start_probs = F.softmax(start_logits, dim=1)
        start_emb = start_probs.unsqueeze(-1) * board_tokens
        start_context = start_emb.sum(dim=1).unsqueeze(1)

        end_input = self.context_norm(board_tokens + start_context)
        end_logits = self.end_head(end_input).squeeze(-1)
        return start_logits, end_logits
