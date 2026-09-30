import chess.pgn
import csv
import os

def generate_training_data(pgn_filepath, output_csv, target_username="brahimkhattara"):
    if not os.path.exists(pgn_filepath):
        print(f"Error: Could not find {pgn_filepath}")
        return

    print(f"Extracting board states and targets for {target_username}...")
    
    # We use a CSV to store the Input (FEN) and Target (UCI Move)
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Board_State_FEN", "Brahim_Move_UCI"]) # Headers
        
        games_processed = 0
        moves_extracted = 0

        with open(pgn_filepath, "r", encoding="utf-8") as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                
                # Determine what color you were playing
                white_player = game.headers.get("White", "").lower()
                black_player = game.headers.get("Black", "").lower()
                
                if target_username.lower() in white_player:
                    target_color = chess.WHITE
                elif target_username.lower() in black_player:
                    target_color = chess.BLACK
                else:
                    continue # You weren't in this game somehow, skip it.

                # Set up a virtual board to track the state as we iterate through moves
                board = game.board()
                
                for move in game.mainline_moves():
                    # If it is YOUR turn, take a snapshot BEFORE you move
                    if board.turn == target_color:
                        # 1. The Input: Current board state
                        fen_state = board.fen()
                        # 2. The Target: The move you made
                        uci_move = move.uci()
                        
                        writer.writerow([fen_state, uci_move])
                        moves_extracted += 1
                        
                    # Push the move to update the board state for the next iteration
                    board.push(move)
                
                games_processed += 1
                if games_processed % 500 == 0:
                    print(f"  ...Processed {games_processed} games. Extracted {moves_extracted} target moves...")

    print("\n--- Extraction Complete ---")
    print(f"Total pure 10-minute games processed: {games_processed}")
    print(f"Total training parameters (moves) extracted: {moves_extracted}")
    print(f"Your model's brain fuel is ready at: {output_csv}")

# Point this to whatever your 10-minute game file is named in the normalized folder
target_file = "normalized_games/brahim_tc_600.pgn" 
generate_training_data(target_file, "brahim_10min_training_data.csv")