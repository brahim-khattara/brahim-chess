import os
import math
from typing import List, Tuple

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import chess.svg

# Added the period '.' for empty squares
FEN_CHARS = "PNBRQKpnbrqk1234567890wb -/abcdefgh]."
UNIQUE_CHARS = sorted(list(set(FEN_CHARS)))

VOCAB = {char: idx + 1 for idx, char in enumerate(UNIQUE_CHARS)}
PAD_IDX = 0
MAX_FEN_LENGTH = 120


def square_to_index(square_str: str) -> int:
    file_char = square_str[0]
    rank_char = square_str[1]
    file_idx = ord(file_char) - ord("a")
    rank_idx = int(rank_char) - 1
    return rank_idx * 8 + file_idx


def index_to_square(square_idx: int) -> str:
    return chess.square_name(square_idx)


def make_uci(start_idx: int, end_idx: int) -> str:
    return f"{index_to_square(start_idx)}{index_to_square(end_idx)}"


def legal_move_mask(fen_str: str) -> torch.Tensor:
    board = chess.Board(fen_str)
    mask = torch.zeros(4096, dtype=torch.float32)
    for move in board.legal_moves:
        flat_idx = (move.from_square * 64) + move.to_square
        mask[flat_idx] = 1.0
    return mask


def unpack_fen(fen_str: str, last_move_uci: str = "none") -> str:
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


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int = 512, max_len: int = MAX_FEN_LENGTH):
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
    def __init__(self, vocab_size: int = len(VOCAB) + 1, d_model: int = 512, nhead: int = 8, num_layers: int = 6):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_encoder = PositionalEncoding2D(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2048, dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.context_norm = nn.LayerNorm(d_model)

        self.start_head = nn.Linear(d_model, 1)
        self.end_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        memory = self.transformer_encoder(x)

        board_tokens = memory[:, :64, :]

        start_logits = self.start_head(board_tokens).squeeze(-1)
        start_probs = F.softmax(start_logits, dim=1)

        start_emb = start_probs.unsqueeze(-1) * board_tokens
        start_context = start_emb.sum(dim=1).unsqueeze(1)

        end_input = board_tokens + start_context
        end_input = self.context_norm(end_input)
        end_logits = self.end_head(end_input).squeeze(-1)

        return start_logits, end_logits


@st.cache_resource
def load_model(weights_path: str, device: torch.device) -> BrahimClone2D:
    model = BrahimClone2D().to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_data
def load_dataset(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def tokenize_fen(fen_str: str) -> torch.Tensor:
    input_text = unpack_fen(fen_str)
    tokens = [VOCAB.get(c, PAD_IDX) for c in input_text]
    if len(tokens) < MAX_FEN_LENGTH:
        tokens += [PAD_IDX] * (MAX_FEN_LENGTH - len(tokens))
    else:
        tokens = tokens[:MAX_FEN_LENGTH]
    return torch.tensor(tokens, dtype=torch.long)


def get_topk_predictions(
    model: BrahimClone2D,
    fen_str: str,
    device: torch.device,
    k: int = 5,
    apply_mask: bool = True,
) -> List[Tuple[str, float]]:
    input_ids = tokenize_fen(fen_str).unsqueeze(0).to(device)

    with torch.no_grad():
        start_logits, end_logits = model(input_ids)
        start_probs = F.softmax(start_logits, dim=1).squeeze(0)
        end_probs = F.softmax(end_logits, dim=1).squeeze(0)

        joint = torch.outer(start_probs, end_probs).reshape(-1)

        if apply_mask:
            mask = legal_move_mask(fen_str).to(joint.device)
            joint = joint * mask

        topk = torch.topk(joint, k=k)
        results = []
        for flat_idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
            start_idx = flat_idx // 64
            end_idx = flat_idx % 64
            results.append((make_uci(start_idx, end_idx), float(score)))

        return results


def render_board(fen_str: str, predicted_uci: str, actual_uci: str) -> str:
    board = chess.Board(fen_str)
    arrows = []

    try:
        pred_move = chess.Move.from_uci(predicted_uci)
        arrows.append(chess.svg.Arrow(pred_move.from_square, pred_move.to_square, color="#d62728"))
    except ValueError:
        pred_move = None

    if actual_uci:
        try:
            actual_move = chess.Move.from_uci(actual_uci)
            arrows.append(chess.svg.Arrow(actual_move.from_square, actual_move.to_square, color="#2ca02c"))
        except ValueError:
            actual_move = None

    return chess.svg.board(board=board, arrows=arrows, size=400, coordinates=True)


def main() -> None:
    st.set_page_config(page_title="EXP-12 Top-5 Inspector", layout="wide")

    st.title("EXP-12 Top-5 Inspector")

    dataset_options = {
        "Train (elo1100)": os.path.join("data", "splits", "brahim_10min_elo1100_training_set.csv"),
        "Validation (elo1100)": os.path.join("data", "splits", "brahim_10min_elo1100_validation_set.csv"),
        "Test (elo1100)": os.path.join("data", "splits", "brahim_10min_elo1100_test_set.csv"),
    }

    weights_path = st.sidebar.text_input(
        "Model weights path",
        value=os.path.join("models", "brahim_clone_exp12_best.pth"),
    )
    dataset_label = st.sidebar.selectbox("Dataset", list(dataset_options.keys()))
    dataset_path = dataset_options[dataset_label]

    apply_mask = st.sidebar.checkbox("Apply legal move mask", value=True)
    topk = st.sidebar.slider("Top-K", min_value=1, max_value=10, value=5, step=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(weights_path):
        st.error(f"Weights not found: {weights_path}")
        return

    if not os.path.exists(dataset_path):
        st.error(f"Dataset not found: {dataset_path}")
        return

    df = load_dataset(dataset_path)
    max_idx = len(df) - 1

    st.sidebar.markdown(f"Rows: {len(df)}")
    row_idx = st.sidebar.number_input("Row index", min_value=0, max_value=max_idx, value=0, step=1)
    random_btn = st.sidebar.button("Random example")

    if random_btn:
        row_idx = int(torch.randint(0, len(df), (1,)).item())

    row = df.iloc[int(row_idx)]
    fen_str = row["Board_State_FEN"]
    actual_uci = row["Brahim_Move_UCI"]
    actual_short = actual_uci[:4]

    model = load_model(weights_path, device)

    predictions = get_topk_predictions(
        model=model,
        fen_str=fen_str,
        device=device,
        k=topk,
        apply_mask=apply_mask,
    )

    top1_uci = predictions[0][0] if predictions else ""
    board_svg = render_board(fen_str, top1_uci, actual_short)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Position")
        st.markdown(board_svg, unsafe_allow_html=True)
        st.caption("Red: predicted top-1 | Green: actual move")

    with col2:
        st.subheader("Moves")
        st.write(f"**Actual move:** {actual_short}")
        st.write(f"**Top-1 predicted:** {top1_uci}")
        st.write("**Top-K list:**")

        table_rows = []
        for i, (uci, score) in enumerate(predictions, start=1):
            table_rows.append({"rank": i, "uci": uci, "score": score})

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        st.subheader("FEN")
        st.code(fen_str)


if __name__ == "__main__":
    main()
