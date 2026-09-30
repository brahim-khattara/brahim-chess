# Data

| Path | Contents |
|---|---|
| `raw/` | Full Chess.com + Lichess PGN dumps for `brahimkhattara` |
| `normalized/normalized_games/` | Same games bucketed by time control (`brahim_tc_600.pgn` = 10+0) |
| `splits/` | CSV rows of `Board_State_FEN`, `Brahim_Move_UCI`, optional `Opponent_Last_Move_UCI` |

Rebuild with:

```bash
python scripts/data/scrape_chesscom.py
python scripts/data/scrape_lichess.py
python scripts/data/normalize_games.py
python scripts/data/generate_dataset.py --min-elo 1400
python scripts/data/split_dataset.py --prefix brahim_with_history_10min_elo1400
```

Only the author's own moves are extracted (perspective-filtered by username).
