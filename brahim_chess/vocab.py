"""Deterministic character-level vocabularies for FEN tokenization.

Early experiments tokenized raw FEN strings. Later 2D experiments unpack the
board into 64 fixed squares and append move history, so they need a wider
character set. Vocabs are always built from a sorted character list so that
saved weight matrices stay aligned across processes (Python's set() hash
randomization previously broke inference).
"""

from __future__ import annotations

# Raw FEN tokenizer used by EXP-03 / EXP-04 / EXP-07 / EXP-08.
RAW_FEN_CHARS = "PNBRQKpnbrqk12345678wb -/"

# Unpacked-board tokenizer used by EXP-12 / EXP-14 / EXP-15.
# Includes '.' for empty squares, digits 9–0, files a–h, and ']' history sentinel.
UNPACKED_FEN_CHARS = "PNBRQKpnbrqk1234567890wb -/abcdefgh]."

PAD_IDX = 0


def build_vocab(chars: str) -> dict[str, int]:
    unique = sorted(set(chars))
    return {char: idx + 1 for idx, char in enumerate(unique)}


RAW_VOCAB = build_vocab(RAW_FEN_CHARS)
UNPACKED_VOCAB = build_vocab(UNPACKED_FEN_CHARS)

RAW_MAX_LEN = 100
UNPACKED_MAX_LEN = 120
