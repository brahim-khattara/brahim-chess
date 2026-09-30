"""Extract FEN → UCI training rows from a PGN, optionally filtering by Elo."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import chess
import chess.pgn

from brahim_chess.paths import repo_path


def safe_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def generate_training_data(
    pgn_filepath: Path,
    output_csv: Path,
    target_username: str = "brahimkhattara",
    min_elo: int | None = None,
    with_history: bool = True,
) -> None:
    if not pgn_filepath.exists():
        print(f"Error: Could not find {pgn_filepath}")
        return

    elo_msg = f"Elo > {min_elo}" if min_elo is not None else "no Elo filter"
    print(f"Extracting board states for {target_username} ({elo_msg})...")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    header = ["Board_State_FEN", "Brahim_Move_UCI"]
    if with_history:
        header.append("Opponent_Last_Move_UCI")

    games_processed = games_kept = games_skipped = moves_extracted = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        with open(pgn_filepath, "r", encoding="utf-8") as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                white_player = game.headers.get("White", "").lower()
                black_player = game.headers.get("Black", "").lower()

                if target_username.lower() in white_player:
                    target_color = chess.WHITE
                    player_elo = safe_int(game.headers.get("WhiteElo"))
                elif target_username.lower() in black_player:
                    target_color = chess.BLACK
                    player_elo = safe_int(game.headers.get("BlackElo"))
                else:
                    games_skipped += 1
                    continue

                games_processed += 1
                if min_elo is not None and (player_elo is None or player_elo <= min_elo):
                    games_skipped += 1
                    continue

                board = game.board()
                prev_move_uci = ""
                for move in game.mainline_moves():
                    if board.turn == target_color:
                        row = [board.fen(), move.uci()]
                        if with_history:
                            row.append(prev_move_uci)
                        writer.writerow(row)
                        moves_extracted += 1
                    board.push(move)
                    prev_move_uci = move.uci()

                games_kept += 1
                if games_processed % 500 == 0:
                    print(
                        f"  ...Processed {games_processed} games. "
                        f"Kept {games_kept}, skipped {games_skipped}. Moves: {moves_extracted}"
                    )

    print("\n--- Extraction Complete ---")
    print(f"Total games processed: {games_processed}")
    print(f"Games kept: {games_kept}")
    print(f"Games skipped: {games_skipped}")
    print(f"Total moves extracted: {moves_extracted}")
    print(f"Output CSV: {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pgn",
        type=Path,
        default=repo_path("data", "normalized", "normalized_games", "brahim_tc_600.pgn"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_path("data", "splits", "brahim_10min_elo1400_training_data.csv"),
    )
    parser.add_argument("--username", default="brahimkhattara")
    parser.add_argument("--min-elo", type=int, default=1400)
    parser.add_argument("--no-elo-filter", action="store_true")
    parser.add_argument("--no-history", action="store_true")
    args = parser.parse_args()

    generate_training_data(
        args.pgn,
        args.output,
        target_username=args.username,
        min_elo=None if args.no_elo_filter else args.min_elo,
        with_history=not args.no_history,
    )


if __name__ == "__main__":
    main()
