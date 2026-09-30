import os
import math
from typing import Tuple

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


def tokenize_fen(fen_str: str) -> torch.Tensor:
    input_text = unpack_fen(fen_str)
    tokens = [VOCAB.get(c, PAD_IDX) for c in input_text]
    if len(tokens) < MAX_FEN_LENGTH:
        tokens += [PAD_IDX] * (MAX_FEN_LENGTH - len(tokens))
    else:
        tokens = tokens[:MAX_FEN_LENGTH]
    return torch.tensor(tokens, dtype=torch.long)


def pick_model_move(model: BrahimClone2D, board: chess.Board, device: torch.device) -> Tuple[chess.Move, float]:
    fen = board.fen()
    input_ids = tokenize_fen(fen).unsqueeze(0).to(device)

    with torch.no_grad():
        start_logits, end_logits = model(input_ids)
        start_probs = F.softmax(start_logits[0], dim=0).cpu().numpy()
        end_probs = F.softmax(end_logits[0], dim=0).cpu().numpy()

    legal_moves = list(board.legal_moves)
    move_scores = []

    for move in legal_moves:
        uci = move.uci()
        is_promotion = len(uci) == 5

        start_sq = uci[:2]
        end_sq = uci[2:4]
        s_idx = square_to_index(start_sq)
        e_idx = square_to_index(end_sq)

        joint_prob = float(start_probs[s_idx] * end_probs[e_idx])

        if is_promotion and uci[4] != "q":
            joint_prob = 0.0

        move_scores.append((joint_prob, move))

    move_scores.sort(key=lambda x: x[0], reverse=True)
    best_prob, best_move = move_scores[0]
    return best_move, best_prob * 100.0


def render_board_svg(board: chess.Board, last_move: chess.Move | None) -> str:
    arrows = []
    if last_move:
        arrows.append(chess.svg.Arrow(last_move.from_square, last_move.to_square, color="#d62728"))

    return chess.svg.board(board=board, arrows=arrows, size=420, coordinates=True)


def reset_game() -> None:
    st.session_state.board = chess.Board()
    st.session_state.move_history = []
    st.session_state.last_move = None


def main() -> None:
    st.set_page_config(page_title="Play Brahim Clone (EXP-12)", layout="wide")
    st.title("Play Brahim Clone (EXP-12)")

    weights_path = st.sidebar.text_input(
        "Model weights path",
        value=os.path.join("models", "brahim_clone_exp15_finetuned_best.pth"),
    )
    player_color = st.sidebar.selectbox("You play", ["White", "Black"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "board" not in st.session_state:
        reset_game()

    if st.sidebar.button("New game"):
        reset_game()

    if not os.path.exists(weights_path):
        st.error(f"Weights not found: {weights_path}")
        return

    model = load_model(weights_path, device)
    board = st.session_state.board

    player_is_white = player_color == "White"
    model_turn = (board.turn == chess.WHITE and not player_is_white) or (
        board.turn == chess.BLACK and player_is_white
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Board")
        board_svg = render_board_svg(board, st.session_state.last_move)
        st.markdown(board_svg, unsafe_allow_html=True)
        st.caption("Red arrow shows the last move played.")

    with col2:
        st.subheader("Controls")
        st.write(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")

        if board.is_game_over():
            st.success(f"Game over: {board.result()} | {board.outcome().termination.name}")
        elif model_turn:
            st.info("Model is thinking...")
            model_move, confidence = pick_model_move(model, board, device)
            board.push(model_move)
            st.session_state.last_move = model_move
            st.session_state.move_history.append(
                f"Model: {model_move.uci()} ({confidence:.2f}%)"
            )
            st.rerun()
        else:
            with st.form("player_move"):
                move_text = st.text_input("Your move (UCI)", value="", placeholder="e2e4")
                submitted = st.form_submit_button("Play move")

            if submitted:
                move_text = move_text.strip().lower()
                try:
                    move = chess.Move.from_uci(move_text)
                except ValueError:
                    st.error("Invalid UCI format.")
                else:
                    if move in board.legal_moves:
                        board.push(move)
                        st.session_state.last_move = move
                        st.session_state.move_history.append(f"You: {move_text}")
                        st.rerun()
                    else:
                        st.error("Illegal move for this position.")

        st.subheader("Move log")
        if st.session_state.move_history:
            st.text("\n".join(st.session_state.move_history[-20:]))
        else:
            st.caption("No moves yet.")


if __name__ == "__main__":
    main()
