# Project report — BrahimClone

## Objective

Build a neural net that behaves like a specific human chess player (Brahim), trained only on the author’s 10-minute rapid games. Target = stylistic fidelity, not Stockfish strength.

## Metrics

- **Unmasked exact** — argmax over all start×end pairs
- **Masked Top-1** — illegal moves zeroed first (fair identity score)
- **Masked Top-5** — human move in the top-5 legal candidates (style / intent score)

## Results

| Exp | Idea | Masked Top-1 | Masked Top-5 |
|---|---|---|---|
| 01 | Baseline, all 10-min | 8.68% *(unmasked)* | — |
| 03 | Elo ≥ 1100 + legal mask | **18.50%** | — |
| 04 | Coupled heads | 17.92% | — |
| 07 | 4096-way action head | ~2% *(failed)* | — |
| 08 | History on raw FEN | ~0–2% *(failed)* | — |
| **12** | Unpacked FEN + 2D PE | **42.66%** | **72.66%** |
| 13 | Spatial CNN | ~18% unmasked val *(overfit)* | — |
| 14 | 2D + history, Elo ≥ 1400 | 33.36% | 59.88% |
| 15 | Fine-tune EXP-12 → Elo ≥ 1400 | 31.04% | **63.22%** |

EXP-12/14/15 measured with `experiments/eval_topk.py` on held-out test splits.

## Critical bug

`set(FEN_CHARS)` iteration is process-randomized. Vocabs must be `sorted(set(...))` or inference silently scores ~0.12%.

## Production model

`BrahimClone2D` — unpacked 64-square FEN, 2D file/rank PE, per-square start/end heads. Default play weights: `brahim_clone_exp15_finetuned_best.pth`.
