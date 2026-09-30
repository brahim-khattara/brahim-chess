"""Streamlit app — play a game against the BrahimClone2D model."""

from __future__ import annotations

import chess
import chess.svg
import streamlit as st
import torch

from brahim_chess.inference import get_device, load_brahim_clone_2d, predict_move
from brahim_chess.paths import repo_path


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
    st.set_page_config(page_title="Play BrahimClone", layout="wide")
    st.title("Play BrahimClone")
    st.caption("A Transformer trained to imitate Brahim's 10-minute rapid style.")

    weights_path = st.sidebar.text_input(
        "Model weights path",
        value=str(repo_path("models", "brahim_clone_exp15_finetuned_best.pth")),
    )
    player_color = st.sidebar.selectbox("You play", ["White", "Black"])
    device = get_device()

    if "board" not in st.session_state:
        reset_game()
    if st.sidebar.button("New game"):
        reset_game()

    if not torch.cuda.is_available() and device.type == "cpu":
        st.sidebar.info("Running on CPU — moves may be a bit slow.")

    if not __import__("os").path.exists(weights_path):
        st.error(f"Weights not found: {weights_path}")
        st.info("Train a model first, or place a checkpoint under `models/`.")
        return

    @st.cache_resource
    def _load(path: str, device_str: str):
        return load_brahim_clone_2d(path, torch.device(device_str))

    model = _load(weights_path, str(device))
    board = st.session_state.board
    player_is_white = player_color == "White"
    model_turn = (board.turn == chess.WHITE and not player_is_white) or (
        board.turn == chess.BLACK and player_is_white
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Board")
        st.markdown(render_board_svg(board, st.session_state.last_move), unsafe_allow_html=True)
        st.caption("Red arrow shows the last move played.")

    with col2:
        st.subheader("Controls")
        st.write(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")

        if board.is_game_over():
            outcome = board.outcome()
            term = outcome.termination.name if outcome else "unknown"
            st.success(f"Game over: {board.result()} | {term}")
        elif model_turn:
            st.info("Model is thinking...")
            last_uci = st.session_state.last_move.uci() if st.session_state.last_move else "none"
            model_move, confidence = predict_move(model, board, device, last_move_uci=last_uci)
            board.push(model_move)
            st.session_state.last_move = model_move
            st.session_state.move_history.append(f"Model: {model_move.uci()} ({confidence:.2f}%)")
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
