"""FEN unpacking and tokenization."""

from __future__ import annotations

import torch

from brahim_chess.vocab import PAD_IDX, RAW_MAX_LEN, RAW_VOCAB, UNPACKED_MAX_LEN, UNPACKED_VOCAB


def unpack_fen(fen_str: str, last_move_uci: str = "none") -> str:
    """Expand FEN board ranks into 64 fixed chars (a1→h8) plus metadata + history.

    Ranks are reversed so index 0 is a1, matching chess.square indexing used by
    the per-square prediction heads.
    """
    parts = fen_str.split(" ")
    board = parts[0]
    metadata = " ".join(parts[1:])

    ranks = board.split("/")
    ranks.reverse()

    unpacked = ""
    for rank in ranks:
        for char in rank:
            if char.isdigit():
                unpacked += "." * int(char)
            else:
                unpacked += char

    last_move = last_move_uci if isinstance(last_move_uci, str) else "none"
    padded_history = last_move.strip().ljust(4, " ") + "]"
    return f"{unpacked} {metadata} {padded_history}"


def tokenize_text(
    text: str,
    vocab: dict[str, int] | None = None,
    max_len: int | None = None,
) -> torch.Tensor:
    vocab = vocab if vocab is not None else UNPACKED_VOCAB
    max_len = max_len if max_len is not None else UNPACKED_MAX_LEN
    tokens = [vocab.get(c, PAD_IDX) for c in text]
    if len(tokens) < max_len:
        tokens += [PAD_IDX] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return torch.tensor(tokens, dtype=torch.long)


def tokenize_raw_fen(fen_str: str) -> torch.Tensor:
    return tokenize_text(fen_str, vocab=RAW_VOCAB, max_len=RAW_MAX_LEN)


def tokenize_unpacked_fen(fen_str: str, last_move_uci: str = "none") -> torch.Tensor:
    return tokenize_text(unpack_fen(fen_str, last_move_uci))
