from pathlib import Path

import pandas as pd

from project_paths import RAW_DATA_DIR


DATA_PATH = RAW_DATA_DIR / "train.csv"

print("=" * 50)
print("Loading raw training data...")
print("=" * 50)

# Read the original dataset from the classified raw-data folder.
df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print("=" * 50)

print("\nFirst 5 rows:")
print(df.head())
print("=" * 50)

print("\nColumns:")
print(df.columns.tolist())
print("=" * 50)

print("\nData info:")
df.info()
print("=" * 50)

print("\nNumeric summary:")
print(df.describe())
print("=" * 50)

print("\nData exploration completed.")
