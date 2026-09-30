# BrahimClone

**A Transformer that plays chess like me — not like Stockfish.**

BrahimClone is a behavioral-cloning project: scrape my Chess.com / Lichess games, train a neural net exclusively on *my* 10-minute rapid moves, and measure how well it reproduces my style. The goal is stylistic fidelity, not perfect play.

<p align="center">
  <img src="plots/exp04_exact_accuracy.png" width="720" alt="EXP-04 exact accuracy curves" />
</p>

---

## Highlights

- End-to-end pipeline: scrape → normalize by time control → FEN datasets → train → evaluate → play
- Controlled experiment ladder (Elo filters, legal-move masking, coupled heads, unified action space, 2D positional encodings, CNN baseline, fine-tuning)
- Production architecture: **6-layer Transformer** with **2D board positional encodings** and per-square start/end heads
- Interactive Streamlit apps to play the clone and inspect Top-5 predictions
- Deterministic FEN tokenizer (fixes a real Python `set()` hash-randomization bug that silently zeroed accuracy at inference)

## Results (masked exact move accuracy)

Legal-move masking at eval time (illegal `(from, to)` pairs zeroed via `python-chess`) is the fair comparison — it measures whether the model's *intent* matches the human move among legal options.

| Experiment | Idea | Masked exact |
|---|---|---|
| EXP-01 | Baseline Transformer, all 10-min games, no masking | 8.68% *(unmasked)* |
| EXP-03 | Elo ≥ 1100 + forced legality at eval | **18.50%** |
| EXP-04 | Soft-coupled start→end heads | 17.92% |
| EXP-07 | Unified 4096-way action head | see `plots/exp07_*` |
| EXP-08 | Opponent last-move history prefix | see `plots/exp08_*` |
| EXP-12 | Unpacked FEN + 2D positional encodings | see `plots/exp12_*` |
| EXP-13 | Spatial ResNet CNN (14×8×8) | see `plots/exp13_*` |
| EXP-14 / 15 | History + Elo ≥ 1400 / fine-tune from EXP-12 | see `plots/exp14_*`, `plots/exp15_*` |

Qualitative boards (green = human, red = model) live under `plots/*/`.

---

## Project layout

```
brahim-chess/
├── brahim_chess/          # Shared library (models, data, metrics, inference)
├── apps/                  # Streamlit demos
│   ├── play.py            # Play against the clone
│   └── inspect_top5.py    # Top-5 move inspector
├── experiments/
│   ├── train_2d.py        # Production trainer (EXP-12 / 14 / 15)
│   ├── eval_topk.py       # Masked Top-1 / Top-5 eval
│   └── legacy/            # Original self-contained experiment scripts
├── scripts/data/          # Scrape → normalize → CSV → split
├── data/                  # PGNs + CSV splits (see data/README.md)
├── models/                # Checkpoints (gitignored — train locally)
├── plots/                 # Training curves + qualitative SVGs
└── docs/                  # Experiment notes
```

## Quickstart

```bash
# 1. Environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt

# 2. (Optional) rebuild datasets from your own games
python scripts/data/scrape_chesscom.py --username YOUR_USER
python scripts/data/scrape_lichess.py --username YOUR_USER
python scripts/data/normalize_games.py
python scripts/data/generate_dataset.py --min-elo 1400
python scripts/data/split_dataset.py

# 3. Train the production 2D model (fine-tune path = EXP-15)
#    Requires an EXP-12 checkpoint, or pass --no-init to train from scratch.
python experiments/train_2d.py --exp-name exp15_finetuned

# 4. Evaluate Top-1 / Top-5
python experiments/eval_topk.py

# 5. Play
streamlit run apps/play.py
```

> **Note:** Model weights (`.pth`) are not committed — they are ~70MB each and total over 1GB. Train locally or ask me for a release artifact.

## Architecture (production)

1. **Unpack** the FEN board into 64 fixed characters (`a1`→`h8`) so token index = chess square index
2. Append FEN metadata + a padded opponent last-move slot
3. Embed with a **deterministic** character vocab
4. Add **2D positional encodings** (learned file + rank embeddings) on the 64 board tokens
5. Run a 6-layer Transformer encoder (`d_model=512`, 8 heads)
6. Predict **start square** with a per-token linear head, then condition the **end square** head on a soft start context
7. At inference, score only **legal** moves

## Key engineering lesson

During Top-5 evaluation, a saved checkpoint scored **0.12%** in a fresh process. Root cause: `VOCAB = {c: i for i, c in enumerate(set(FEN_CHARS))}` — Python randomizes `set()` iteration order per process, so embedding rows silently remapped. Fix: always build vocabs from `sorted(set(...))`.

## Roadmap

- [ ] Publish best checkpoint via GitHub Releases
- [ ] Lichess Bot API live deployment
- [ ] Wins-only dataset ablation (EXP-05)
- [ ] Stronger temporal context (multi-move history / clock features)

## License

MIT — see [LICENSE](LICENSE).
