import chess.pgn
import csv
import os


def safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def generate_training_data_with_elo_filter(
    pgn_filepath,
    output_csv,
    target_username="brahimkhattara",
    min_elo=1400,
):
    if not os.path.exists(pgn_filepath):
        print(f"Error: Could not find {pgn_filepath}")
        return

    print(f"Extracting board states for {target_username} (Elo > {min_elo})...")

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Board_State_FEN", "Brahim_Move_UCI", "Opponent_Last_Move_UCI"])

        games_processed = 0
        games_kept = 0
        games_skipped = 0
        moves_extracted = 0

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

                if player_elo is None or player_elo <= min_elo:
                    games_skipped += 1
                    continue

                board = game.board()
                prev_move_uci = ""
                for move in game.mainline_moves():
                    if board.turn == target_color:
                        fen_state = board.fen()
                        uci_move = move.uci()
                        writer.writerow([fen_state, uci_move, prev_move_uci])
                        moves_extracted += 1
                    board.push(move)
                    prev_move_uci = move.uci()

                games_kept += 1

                if games_processed % 500 == 0:
                    print(
                        f"  ...Processed {games_processed} games. "
                        f"Kept {games_kept}, skipped {games_skipped}. "
                        f"Moves: {moves_extracted}"
                    )

    print("\n--- Extraction Complete ---")
    print(f"Total games processed: {games_processed}")
    print(f"Games kept (Elo > {min_elo}): {games_kept}")
    print(f"Games skipped: {games_skipped}")
    print(f"Total moves extracted: {moves_extracted}")
    print(f"Output CSV: {output_csv}")


if __name__ == "__main__":
    input_pgn = os.path.join("data", "normalized", "normalized_games", "brahim_tc_600.pgn")
    output_csv = os.path.join("data", "splits", "brahim_10min_elo1400_training_data.csv")

    generate_training_data_with_elo_filter(input_pgn, output_csv)
