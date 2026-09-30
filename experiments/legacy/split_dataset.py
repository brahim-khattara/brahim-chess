import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv("brahim_10min_training_data.csv")
print(f"Total rows: {len(df)}")

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split: 5k test, 5k validation, rest training
test_size = 5000
val_size = 5000

test_df = df[:test_size]
val_df = df[test_size:test_size + val_size]
train_df = df[test_size + val_size:]

print(f"\nDataset Split:")
print(f"Training set: {len(train_df)} rows")
print(f"Validation set: {len(val_df)} rows")
print(f"Test set: {len(test_df)} rows")

# Save the splits
train_df.to_csv("brahim_10min_training_set.csv", index=False)
val_df.to_csv("brahim_10min_validation_set.csv", index=False)
test_df.to_csv("brahim_10min_test_set.csv", index=False)

print("\nFiles saved:")
print("- brahim_10min_training_set.csv")
print("- brahim_10min_validation_set.csv")
print("- brahim_10min_test_set.csv")
