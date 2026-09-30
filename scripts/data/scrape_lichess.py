"""Stream all public Lichess games for a username into a PGN file."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests

from brahim_chess.paths import repo_path


def scrape_lichess_games(username: str, output_path: Path) -> None:
    headers = {
        "Accept": "application/x-chess-pgn",
        "User-Agent": f"BrahimClone-DataScraper (Contact: {username}@lichess.org)",
    }
    params = {"evals": "true", "clocks": "true", "opening": "true"}
    url = f"https://lichess.org/api/games/user/{username}"

    print(f"Opening Lichess data stream for: {username}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with requests.get(url, headers=headers, params=params, stream=True, timeout=30) as response:
            response.raise_for_status()
            with open(output_path, "w", encoding="utf-8") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk.decode("utf-8"))
        print(f"\nExtraction complete! All Lichess games saved to '{output_path}'.")
    except requests.exceptions.RequestException as exc:
        print(f"\n[!] Stream failed: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", default="brahimkhattara")
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_path("data", "raw", "brahimkhattara_lichess_raw_games.pgn"),
    )
    args = parser.parse_args()
    scrape_lichess_games(args.username, args.output)


if __name__ == "__main__":
    main()
