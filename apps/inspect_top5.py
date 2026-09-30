"""Streamlit inspector — compare the model's Top-5 legal moves to the human move."""

from __future__ import annotations

import chess
import chess.svg
import pandas as pd
import streamlit as st
import torch

from brahim_chess.inference import get_device, load_brahim_clone_2d, predict_topk
from brahim_chess.paths import repo_path


def main() -> None:
    st.set_page_config(page_title="BrahimClone Top-5 Inspector", layout="wide")
    st.title("BrahimClone Top-5 Inspector")
    st.caption("Green = human move · Red = model Top-1")

    weights_path = st.sidebar.text_input(
        "Model weights path",
        value=str(repo_path("models", "brahim_clone_exp15_finetuned_best.pth")),
    )
    csv_path = st.sidebar.text_input(
        "Positions CSV",
        value=str(repo_path("data", "splits", "brahim_with_history_10min_elo1400_test_set.csv")),
    )
    device = get_device()

    if not __import__("os").path.exists(weights_path):
        st.error(f"Weights not found: {weights_path}")
        return
    if not __import__("os").path.exists(csv_path):
        st.error(f"CSV not found: {csv_path}")
        return

    @st.cache_resource
    def _load(path: str, device_str: str):
        return load_brahim_clone_2d(path, torch.device(device_str))

    @st.cache_data
    def _load_csv(path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    model = _load(weights_path, str(device))
    df = _load_csv(csv_path)
    idx = st.sidebar.number_input("Row index", min_value=0, max_value=len(df) - 1, value=0, step=1)
    row = df.iloc[int(idx)]
    fen = row["Board_State_FEN"]
    actual = str(row["Brahim_Move_UCI"])[:4]
    last_move = row.get("Opponent_Last_Move_UCI", "none")
    if pd.isna(last_move) or not last_move:
        last_move = "none"

    ranked = predict_topk(model, fen, device, last_move_uci=str(last_move), k=5)
    board = chess.Board(fen)
    arrows = []
    if ranked:
        try:
            top = chess.Move.from_uci(ranked[0][0])
            arrows.append(chess.svg.Arrow(top.from_square, top.to_square, color="#d62728"))
        except ValueError:
            pass
    try:
        human = chess.Move.from_uci(actual)
        arrows.append(chess.svg.Arrow(human.from_square, human.to_square, color="#2ca02c"))
    except ValueError:
        pass

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Position")
        st.markdown(
            chess.svg.board(board=board, arrows=arrows, size=420, coordinates=True),
            unsafe_allow_html=True,
        )
        st.code(fen)
    with col2:
        st.subheader("Predictions")
        st.write(f"**Human move:** `{actual}`")
        for i, (uci, score) in enumerate(ranked, start=1):
            marker = " ← human" if uci == actual else ""
            st.write(f"{i}. `{uci}` — {score * 100:.2f}%{marker}")


if __name__ == "__main__":
    main()
