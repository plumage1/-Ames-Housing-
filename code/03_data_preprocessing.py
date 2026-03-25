import joblib
import pandas as pd

from preprocessing_utils import DROP_COLS, preprocess_inference_dataframe, preprocess_training_dataframe
from project_paths import DOCS_DIR, OUTPUT_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR


TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"

print("=" * 50)
print("Starting data preprocessing...")
print("=" * 50)

# Ensure output folders exist.
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

train_df = pd.read_csv(TRAIN_PATH)
print("Original train shape:", train_df.shape)

# Fit preprocessing on training data and generate artifacts for reuse.
X_processed, y, summary_df, artifacts = preprocess_training_dataframe(
    train_df,
    drop_cols=DROP_COLS,
)

# Save processed datasets and documentation tables.
train_processed_df = pd.concat([X_processed, y.rename("SalePrice")], axis=1)
X_processed.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
y.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
train_processed_df.to_csv(PROCESSED_DATA_DIR / "train_processed.csv", index=False)
summary_df.to_csv(DOCS_DIR / "preprocessing_summary.csv", index=False)
joblib.dump(artifacts, OUTPUT_DIR / "preprocessing_artifacts.pkl")

print("Processed training features shape:", X_processed.shape)
print("Processed target shape:", y.shape)
print(
    "Total outliers clipped:",
    int(summary_df["outlier_count"].fillna(0).sum()),
)

if TEST_PATH.exists():
    test_df = pd.read_csv(TEST_PATH)
    # Apply the exact same preprocessing rules to the test set.
    X_test_processed = preprocess_inference_dataframe(test_df, artifacts)
    X_test_processed.to_csv(PROCESSED_DATA_DIR / "X_test_processed.csv", index=False)
    print("Processed test features shape:", X_test_processed.shape)

print("Saved processed train features to:", PROCESSED_DATA_DIR / "X_train.csv")
print("Saved processed target to:", PROCESSED_DATA_DIR / "y_train.csv")
print("Saved combined processed dataset to:", PROCESSED_DATA_DIR / "train_processed.csv")
print("Saved preprocessing summary to:", DOCS_DIR / "preprocessing_summary.csv")
print("Saved preprocessing artifacts to:", OUTPUT_DIR / "preprocessing_artifacts.pkl")
print("=" * 50)
print("Data preprocessing completed.")
