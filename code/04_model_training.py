import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from preprocessing_utils import DROP_COLS, preprocess_inference_dataframe, preprocess_training_dataframe
from project_paths import OUTPUT_DIR, RAW_DATA_DIR, RESULT_DIR


TARGET_COL = "SalePrice"
TRAIN_PATH = RAW_DATA_DIR / "train.csv"
MODEL_BUILDERS = {
    "LinearRegression": lambda: LinearRegression(),
    "RandomForest": lambda: RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=1,
    ),
}


def calculate_metrics(model_name, y_true, predictions):
    """Calculate common regression metrics for a prediction vector."""
    mse = mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)
    return {
        "model": model_name,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
    }


def cross_validate_model(model_name, model_builder, train_eval_df, n_splits=5):
    """Run K-fold CV on raw data so preprocessing is fit inside each fold."""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    for fold_id, (fit_idx, valid_idx) in enumerate(kfold.split(train_eval_df), start=1):
        fit_df = train_eval_df.iloc[fit_idx].reset_index(drop=True)
        valid_df = train_eval_df.iloc[valid_idx].reset_index(drop=True)

        X_fit, y_fit, _, artifacts = preprocess_training_dataframe(
            fit_df,
            drop_cols=DROP_COLS,
        )
        X_valid = preprocess_inference_dataframe(
            valid_df.drop(columns=[TARGET_COL]),
            artifacts,
        )
        y_valid = valid_df[TARGET_COL].reset_index(drop=True)

        model = model_builder()
        model.fit(X_fit, y_fit)
        predictions = model.predict(X_valid)
        metrics = calculate_metrics(model_name, y_valid, predictions)
        metrics["fold"] = fold_id
        fold_metrics.append(metrics)

    fold_metrics_df = pd.DataFrame(fold_metrics)
    summary = {
        "model": model_name,
        "cv_MSE_mean": fold_metrics_df["MSE"].mean(),
        "cv_MSE_std": fold_metrics_df["MSE"].std(ddof=0),
        "cv_RMSE_mean": fold_metrics_df["RMSE"].mean(),
        "cv_RMSE_std": fold_metrics_df["RMSE"].std(ddof=0),
        "cv_MAE_mean": fold_metrics_df["MAE"].mean(),
        "cv_MAE_std": fold_metrics_df["MAE"].std(ddof=0),
        "cv_R2_mean": fold_metrics_df["R2"].mean(),
        "cv_R2_std": fold_metrics_df["R2"].std(ddof=0),
    }
    return summary, fold_metrics_df


def evaluate_on_holdout(model_name, model_builder, train_eval_df, holdout_df):
    """Fit preprocessing on the training split and evaluate on untouched holdout data."""
    X_train_eval, y_train_eval, _, artifacts = preprocess_training_dataframe(
        train_eval_df,
        drop_cols=DROP_COLS,
    )
    X_holdout = preprocess_inference_dataframe(
        holdout_df.drop(columns=[TARGET_COL]),
        artifacts,
    )
    y_holdout = holdout_df[TARGET_COL].reset_index(drop=True)

    model = model_builder()
    model.fit(X_train_eval, y_train_eval)
    predictions = model.predict(X_holdout)
    metrics = calculate_metrics(model_name, y_holdout, predictions)
    return metrics, y_holdout, predictions


# Prepare output folders for models and plots.
OUTPUT_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

print("=" * 50)
print("Starting model training...")
print("=" * 50)

# Load the raw training data so preprocessing can be fit without leakage.
train_df = pd.read_csv(TRAIN_PATH)

print("Raw dataset shape:", train_df.shape)

# Keep a final holdout set untouched during model selection.
train_eval_df, holdout_df = train_test_split(
    train_df,
    test_size=0.2,
    random_state=42,
    shuffle=True,
)
train_eval_df = train_eval_df.reset_index(drop=True)
holdout_df = holdout_df.reset_index(drop=True)

print("Model-selection rows:", train_eval_df.shape[0])
print("Holdout rows:", holdout_df.shape[0])

cv_summaries = []
cv_fold_tables = []
holdout_summaries = []
holdout_predictions = {}

for model_name, model_builder in MODEL_BUILDERS.items():
    print("\n" + "-" * 50)
    print(f"[Model] {model_name}")
    print("-" * 50)

    cv_summary, fold_metrics_df = cross_validate_model(
        model_name,
        model_builder,
        train_eval_df,
    )
    cv_summaries.append(cv_summary)

    cv_fold_tables.append(fold_metrics_df.copy())

    holdout_metrics, y_holdout, predictions = evaluate_on_holdout(
        model_name,
        model_builder,
        train_eval_df,
        holdout_df,
    )
    holdout_summary = {
        "model": model_name,
        "holdout_MSE": holdout_metrics["MSE"],
        "holdout_RMSE": holdout_metrics["RMSE"],
        "holdout_MAE": holdout_metrics["MAE"],
        "holdout_R2": holdout_metrics["R2"],
    }
    holdout_summaries.append(holdout_summary)
    holdout_predictions[model_name] = {
        "actual": y_holdout,
        "predicted": pd.Series(predictions),
    }

    print("CV mean metrics:")
    print(f"R2   : {cv_summary['cv_R2_mean']:.4f} +/- {cv_summary['cv_R2_std']:.4f}")
    print(f"RMSE : ${cv_summary['cv_RMSE_mean']:,.2f}")
    print(f"MAE  : ${cv_summary['cv_MAE_mean']:,.2f}")
    print("Holdout metrics:")
    print(f"R2   : {holdout_summary['holdout_R2']:.4f}")
    print(f"RMSE : ${holdout_summary['holdout_RMSE']:,.2f}")
    print(f"MAE  : ${holdout_summary['holdout_MAE']:,.2f}")

metrics_df = pd.DataFrame(cv_summaries).merge(
    pd.DataFrame(holdout_summaries),
    on="model",
).sort_values(by="cv_R2_mean", ascending=False)
metrics_df.to_csv(RESULT_DIR / "model_metrics.csv", index=False)
pd.concat(cv_fold_tables, ignore_index=True).to_csv(
    RESULT_DIR / "cv_fold_metrics.csv",
    index=False,
)

best_row = metrics_df.iloc[0]
best_name = best_row["model"]
best_prediction_bundle = holdout_predictions[best_name]

# Fit preprocessing on the full dataset before saving deployable models.
X_full, y_full, _, full_artifacts = preprocess_training_dataframe(
    train_df,
    drop_cols=DROP_COLS,
)
joblib.dump(full_artifacts, OUTPUT_DIR / "preprocessing_artifacts.pkl")
pd.DataFrame(
    full_artifacts["encoded_feature_columns"],
    columns=["feature"],
).to_csv(OUTPUT_DIR / "feature_columns.csv", index=False)

full_lr = MODEL_BUILDERS["LinearRegression"]().fit(X_full, y_full)
full_rf = MODEL_BUILDERS["RandomForest"]().fit(X_full, y_full)
final_best_model = full_rf if best_name == "RandomForest" else full_lr

joblib.dump(full_lr, OUTPUT_DIR / "LinearRegression_model.pkl")
joblib.dump(full_rf, OUTPUT_DIR / "RandomForest_model.pkl")
joblib.dump(final_best_model, OUTPUT_DIR / "best_model.pkl")
(OUTPUT_DIR / "best_model_name.txt").write_text(str(best_name), encoding="utf-8")

# Build a diagnostics table for evaluation plots.
prediction_df = pd.DataFrame(
    {
        "actual_sale_price": best_prediction_bundle["actual"].reset_index(drop=True),
        "predicted_sale_price": best_prediction_bundle["predicted"].reset_index(drop=True),
    }
)
prediction_df["residual"] = (
    prediction_df["actual_sale_price"] - prediction_df["predicted_sale_price"]
)
prediction_df.to_csv(RESULT_DIR / "test_set_predictions.csv", index=False)

# Plot model comparison metrics.
comparison_fig, comparison_axes = plt.subplots(1, 2, figsize=(10, 4))
comparison_axes[0].bar(
    metrics_df["model"],
    metrics_df["cv_RMSE_mean"],
    color=["#d95f02", "#1b9e77"],
)
comparison_axes[0].set_title("Cross-Validation RMSE")
comparison_axes[0].set_ylabel("RMSE")
comparison_axes[1].bar(
    metrics_df["model"],
    metrics_df["cv_R2_mean"],
    color=["#7570b3", "#66a61e"],
)
comparison_axes[1].set_title("Cross-Validation R2")
comparison_axes[1].set_ylabel("R2")
comparison_fig.tight_layout()
comparison_fig.savefig(RESULT_DIR / "model_comparison.png", dpi=300)
plt.close(comparison_fig)

# Plot actual vs predicted values for the best model.
scatter_fig, scatter_ax = plt.subplots(figsize=(6, 6))
scatter_ax.scatter(
    prediction_df["actual_sale_price"],
    prediction_df["predicted_sale_price"],
    alpha=0.6,
    color="#1f78b4",
)
min_price = min(prediction_df["actual_sale_price"].min(), prediction_df["predicted_sale_price"].min())
max_price = max(prediction_df["actual_sale_price"].max(), prediction_df["predicted_sale_price"].max())
scatter_ax.plot([min_price, max_price], [min_price, max_price], linestyle="--", color="black")
scatter_ax.set_title(f"Actual vs Predicted ({best_name})")
scatter_ax.set_xlabel("Actual SalePrice")
scatter_ax.set_ylabel("Predicted SalePrice")
scatter_fig.tight_layout()
scatter_fig.savefig(RESULT_DIR / "actual_vs_predicted.png", dpi=300)
plt.close(scatter_fig)

# Plot residual distribution.
residual_fig, residual_ax = plt.subplots(figsize=(7, 4))
residual_ax.hist(prediction_df["residual"], bins=30, color="#e7298a", edgecolor="black")
residual_ax.set_title(f"Residual Distribution ({best_name})")
residual_ax.set_xlabel("Residual")
residual_ax.set_ylabel("Count")
residual_fig.tight_layout()
residual_fig.savefig(RESULT_DIR / "residual_distribution.png", dpi=300)
plt.close(residual_fig)

print("\n" + "=" * 50)
print("Model comparison")
print("=" * 50)
print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
print(f"\nBest model: {best_name} (CV R2 = {best_row['cv_R2_mean']:.4f})")
print("Saved metrics to:", RESULT_DIR / "model_metrics.csv")
print("Saved fold-by-fold CV metrics to:", RESULT_DIR / "cv_fold_metrics.csv")
print("Saved prediction diagnostics to result folder.")
print("Saved models and feature columns to output folder.")
print("=" * 50)
print("Model training completed.")
