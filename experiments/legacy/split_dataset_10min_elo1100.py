import os

import numpy as np
import pandas as pd

np.random.seed(42)


def split_dataset(input_csv, output_dir, test_size=5000, val_size=5000):
    print("Loading dataset...")
    df = pd.read_csv(input_csv)
    print(f"Total rows: {len(df)}")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    test_df = df[:test_size]
    val_df = df[test_size : test_size + val_size]
    train_df = df[test_size + val_size :]

    print("\nDataset Split:")
    print(f"Training set: {len(train_df)} rows")
    print(f"Validation set: {len(val_df)} rows")
    print(f"Test set: {len(test_df)} rows")

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "brahim_with_history_10min_elo1400_training_set.csv")
    val_path = os.path.join(output_dir, "brahim_with_history_10min_elo1400_validation_set.csv")
    test_path = os.path.join(output_dir, "brahim_with_history_10min_elo1400_test_set.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nFiles saved:")
    print(f"- {train_path}")
    print(f"- {val_path}")
    print(f"- {test_path}")


if __name__ == "__main__":
    input_csv = os.path.join("data", "splits", "brahim_10min_elo1400_training_data.csv")
    split_dataset(input_csv, output_dir=os.path.join("data", "splits"))
