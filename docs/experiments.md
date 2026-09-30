# Project report — BrahimClone

## Objective

Build, train, and evaluate a neural network that behaves like a specific human chess player (Brahim). The model is trained only on the author's historical 10-minute rapid games — the target is stylistic fidelity, not Stockfish-level strength.

Long-term goal: deploy via the Lichess Bot API and measure live rating / qualitative style.

## Pipeline

1. Scrape Chess.com + Lichess PGNs
2. Normalize by time control; keep `tc_600` (10+0)
3. Extract only the author's moves as `(FEN, UCI[, opponent_last_move])`
4. Train Transformer / CNN variants
5. Evaluate with raw and **legality-masked** exact-move accuracy
6. Qualitative board SVGs (human vs model arrows)

## Documented results

| Exp | Change | Result |
|---|---|---|
| 01 | Baseline, all 10-min, no mask | Exact 8.68% |
| 03 | Elo ≥ 1100 + masked eval | Raw 9.76% → **masked 18.50%** |
| 04 | Coupled heads | Raw 9.86% / masked 17.92% |

Later experiments (07–15) log metrics at runtime and write curves under `plots/`.

## Critical bugfix

`set(FEN_CHARS)` iteration order is process-randomized. Building `VOCAB` from an unordered set meant a checkpoint trained in one process could score ~0.12% when loaded in another. Vocabs are now always `sorted(set(...))`.

## Production architecture

`BrahimClone2D`: unpacked 64-square FEN, 2D file/rank positional encodings, per-square start/end heads with soft start→end conditioning. Best weights used by the play app: `brahim_clone_exp15_finetuned_best.pth`.
