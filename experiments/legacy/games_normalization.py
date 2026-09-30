import chess.pgn
import os

def normalize_and_sort_pgns(input_files, output_dir="normalized_games"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    games_processed = 0
    games_skipped = 0
    
    # We will store open file handles in a dictionary based on the time control
    # e.g., "180" for 3-minute, "600" for 10-minute
    output_files = {}

    for filepath in input_files:
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}. Skipping.")
            continue
            
        print(f"Processing {filepath}...")
        
        with open(filepath, "r", encoding="utf-8") as pgn_file:
            while True:
                # Read one game at a time into a structured object
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break # End of file
                
                # 1. Extract and Normalize Time Control
                tc_raw = game.headers.get("TimeControl", "?")
                
                # Filter out correspondence games or missing time controls
                if tc_raw == "?" or tc_raw == "-" or "/" in tc_raw:
                    games_skipped += 1
                    continue
                
                # Lichess uses "180+0", Chess.com uses "180". Let's normalize to base seconds.
                # If there's an increment (e.g., "180+2"), we keep it as a distinct category.
                if "+" in tc_raw:
                    base_time, increment = tc_raw.split("+")
                    if increment == "0":
                        tc_normalized = base_time # Convert "180+0" to "180"
                    else:
                        tc_normalized = f"{base_time}_inc_{increment}"
                else:
                    tc_normalized = tc_raw
                
                # 2. Route the game to the correct file bucket
                if tc_normalized not in output_files:
                    safe_filename = os.path.join(output_dir, f"brahim_tc_{tc_normalized}.pgn")
                    output_files[tc_normalized] = open(safe_filename, "a", encoding="utf-8")
                
                # 3. Write the cleaned game back to the new file
                # The exporter automatically preserves the [%clk ...] comments
                exporter = chess.pgn.FileExporter(output_files[tc_normalized])
                game.accept(exporter)
                
                games_processed += 1
                
                if games_processed % 500 == 0:
                    print(f"  ...Sorted {games_processed} games so far...")

    # Close all open file handles
    for f in output_files.values():
        f.close()

    print("\n--- Normalization Complete ---")
    print(f"Total games successfully sorted: {games_processed}")
    print(f"Total non-standard/daily games skipped: {games_skipped}")
    print(f"Sorted files are located in the '{output_dir}' directory.")

# Define the raw files you downloaded in the previous steps
files_to_process = [
    "brahimkhattara_chess_raw_games.pgn", 
    "brahimkhattara_lichess_raw_games.pgn"
]

normalize_and_sort_pgns(files_to_process)