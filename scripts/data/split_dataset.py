"""Shuffle a training CSV into fixed train / val / test splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from brahim_chess.paths import repo_path


def split_dataset(
    input_csv: Path,
    output_dir: Path,
    prefix: str,
    test_size: int = 5000,
    val_size: int = 5000,
    seed: int = 42,
) -> None:
    np.random.seed(seed)
    print(f"Loading dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"Total rows: {len(df)}")

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = df[:test_size]
    val_df = df[test_size : test_size + val_size]
    train_df = df[test_size + val_size :]

    print("\nDataset Split:")
    print(f"Training set:   {len(train_df)} rows")
    print(f"Validation set: {len(val_df)} rows")
    print(f"Test set:       {len(test_df)} rows")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / f"{prefix}_training_set.csv"
    val_path = output_dir / f"{prefix}_validation_set.csv"
    test_path = output_dir / f"{prefix}_test_set.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nFiles saved:")
    print(f"- {train_path}")
    print(f"- {val_path}")
    print(f"- {test_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_path("data", "splits", "brahim_10min_elo1400_training_data.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=repo_path("data", "splits"))
    parser.add_argument("--prefix", default="brahim_with_history_10min_elo1400")
    parser.add_argument("--test-size", type=int, default=5000)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_dataset(
        args.input,
        args.output_dir,
        prefix=args.prefix,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
