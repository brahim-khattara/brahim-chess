"""Bucket raw PGN dumps by normalized time control."""

from __future__ import annotations

import argparse
from pathlib import Path

import chess.pgn

from brahim_chess.paths import repo_path


def normalize_and_sort_pgns(input_files: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    games_processed = 0
    games_skipped = 0
    output_files: dict[str, object] = {}

    for filepath in input_files:
        if not filepath.exists():
            print(f"File not found: {filepath}. Skipping.")
            continue

        print(f"Processing {filepath}...")
        with open(filepath, "r", encoding="utf-8") as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                tc_raw = game.headers.get("TimeControl", "?")
                if tc_raw in {"?", "-"} or "/" in tc_raw:
                    games_skipped += 1
                    continue

                if "+" in tc_raw:
                    base_time, increment = tc_raw.split("+")
                    tc_normalized = base_time if increment == "0" else f"{base_time}_inc_{increment}"
                else:
                    tc_normalized = tc_raw

                if tc_normalized not in output_files:
                    out_path = output_dir / f"brahim_tc_{tc_normalized}.pgn"
                    output_files[tc_normalized] = open(out_path, "a", encoding="utf-8")

                exporter = chess.pgn.FileExporter(output_files[tc_normalized])
                game.accept(exporter)
                games_processed += 1
                if games_processed % 500 == 0:
                    print(f"  ...Sorted {games_processed} games so far...")

    for handle in output_files.values():
        handle.close()

    print("\n--- Normalization Complete ---")
    print(f"Total games successfully sorted: {games_processed}")
    print(f"Total non-standard/daily games skipped: {games_skipped}")
    print(f"Sorted files are located in '{output_dir}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[
            repo_path("data", "raw", "brahimkhattara_chess_raw_games.pgn"),
            repo_path("data", "raw", "brahimkhattara_lichess_raw_games.pgn"),
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_path("data", "normalized", "normalized_games"),
    )
    args = parser.parse_args()
    normalize_and_sort_pgns(args.inputs, args.output_dir)


if __name__ == "__main__":
    main()
