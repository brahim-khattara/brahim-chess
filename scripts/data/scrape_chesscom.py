"""Download all public Chess.com games for a username as a single PGN file."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests

from brahim_chess.paths import repo_path


def scrape_chesscom_games(username: str, output_path: Path) -> None:
    headers = {"User-Agent": f"BrahimClone-DataScraper (Contact: {username}@chess.com)"}
    archives_url = f"https://api.chess.com/pub/player/{username}/games/archives"
    print(f"Fetching archive index for: {username}...")

    try:
        response = requests.get(archives_url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(f"Failed to fetch archives: {exc}")
        return

    archives = response.json().get("archives", [])
    if not archives:
        print("No games found or invalid username.")
        return

    print(f"Found {len(archives)} months of game data. Starting download...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        for month_url in archives:
            pgn_url = f"{month_url}/pgn"
            print(f"Downloading: {pgn_url}...")
            for attempt in range(3):
                try:
                    pgn_response = requests.get(pgn_url, headers=headers, timeout=15)
                    if pgn_response.status_code == 200:
                        file.write(pgn_response.text)
                        file.write("\n")
                        break
                    print(f"  [!] Attempt {attempt + 1} failed (HTTP {pgn_response.status_code}). Retrying...")
                    time.sleep(2)
                except requests.exceptions.RequestException as exc:
                    print(f"  [!] Attempt {attempt + 1} error: {exc}. Retrying...")
                    time.sleep(2)
            else:
                print(f"  [!!!] Failed to download {pgn_url}. Skipping month.")
            time.sleep(1.5)

    print(f"\nExtraction complete! Data saved to '{output_path}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", default="brahimkhattara")
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_path("data", "raw", "brahimkhattara_chess_raw_games.pgn"),
    )
    args = parser.parse_args()
    scrape_chesscom_games(args.username, args.output)


if __name__ == "__main__":
    main()
