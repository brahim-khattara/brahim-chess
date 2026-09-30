# Experiments

## Modern entry points

| Script | Purpose |
|---|---|
| `train_2d.py` | Train / fine-tune `BrahimClone2D` (EXP-12 / 14 / 15) |
| `eval_topk.py` | Masked Top-1 / Top-5 evaluation |

## Historical ladder (`legacy/`)

Self-contained scripts from the original research process. Kept for reproducibility of older checkpoints and plots; new work should use `brahim_chess/` + the modern entry points above.

| Script | Experiment |
|---|---|
| `train_10min_no_rules.py` | EXP-01 baseline |
| `train_exp03_forced_legality.py` | Elo filter + legal masking |
| `train_exp04_coupled_heads.py` | Coupled heads |
| `train_exp07_unified_action.py` | 4096-way action space |
| `train_exp08_historical_context.py` | History prefix |
| `train_exp12_2d_positional.py` | 2D positional encodings |
| `train_exp13_spatial_cnn.py` | CNN baseline |
| `train_exp14_2d_positional.py` | 2D + history + Elo≥1400 |
| `train_exp12_2d_positional_elo1400.py` | EXP-15 fine-tune |

Run legacy scripts from the **repository root** so relative `data/` / `models/` / `plots/` paths resolve.
