# Models

Trained PyTorch checkpoints (`.pth`) are **not** stored in git — each dual-head Transformer checkpoint is ~70MB and the full experiment suite exceeds 1GB.

Expected filenames after training:

| File | Experiment |
|---|---|
| `brahim_clone_exp12_best.pth` | 2D Transformer on Elo≥1100 |
| `brahim_clone_exp14_best.pth` | 2D + history on Elo≥1400 |
| `brahim_clone_exp15_finetuned_best.pth` | Fine-tune of EXP-12 on Elo≥1400 *(default for apps)* |

Train the production model:

```bash
python experiments/train_2d.py --exp-name exp15_finetuned --no-init
# or fine-tune from an existing EXP-12 checkpoint (default):
python experiments/train_2d.py --exp-name exp15_finetuned
```
