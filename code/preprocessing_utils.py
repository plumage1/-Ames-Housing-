from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


TARGET_COL = "SalePrice"
DROP_COLS = ["Id"]


def _safe_mode(series: pd.Series) -> Any:
    mode = series.mode(dropna=True)
    if not mode.empty:
        return mode.iloc[0]
    return "Missing"


def preprocess_training_dataframe(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    drop_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict[str, Any]]:
    """Fit preprocessing rules on the training dataframe and return scaled features."""
    drop_cols = drop_cols or []

    # Remove non-informative columns (e.g., Id) and split target vs features.
    working_df = df.drop(columns=drop_cols).copy()
    y = working_df[target_col].copy()
    X_raw = working_df.drop(columns=[target_col]).copy()

    # Identify numeric and categorical feature columns.
    numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_raw.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    # Use median for numeric and mode for categorical missing values.
    numeric_fill_values = X_raw[numeric_cols].median().to_dict()
    categorical_fill_values = {
        col: _safe_mode(X_raw[col]) for col in categorical_cols
    }

    missing_counts = X_raw.isnull().sum().to_dict()

    X_filled = X_raw.copy()
    if numeric_cols:
        X_filled[numeric_cols] = X_filled[numeric_cols].fillna(value=numeric_fill_values)
    if categorical_cols:
        X_filled[categorical_cols] = X_filled[categorical_cols].fillna(
            value=categorical_fill_values
        )

    # Clip numeric outliers using the IQR rule.
    X_clipped = X_filled.copy()
    clip_bounds: dict[str, dict[str, float | int]] = {}
    for col in numeric_cols:
        series = X_clipped[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        if pd.isna(lower) or pd.isna(upper):
            lower = series.min()
            upper = series.max()

        outlier_count = int(((series < lower) | (series > upper)).sum())
        X_clipped[col] = series.clip(lower=lower, upper=upper)
        clip_bounds[col] = {
            "lower": float(lower),
            "upper": float(upper),
            "outlier_count": outlier_count,
        }

    # Keep all one-hot columns so unseen categories at inference time map to
    # an all-zero pattern instead of being conflated with a dropped baseline.
    X_encoded = pd.get_dummies(X_clipped, drop_first=False)

    # Standardize features to zero mean / unit variance.
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_encoded),
        columns=X_encoded.columns,
        index=X_encoded.index,
    )

    # Build a readable summary for documentation.
    summary_rows: list[dict[str, Any]] = []
    for col in numeric_cols:
        summary_rows.append(
            {
                "feature": col,
                "type": "numeric",
                "missing_count": int(missing_counts.get(col, 0)),
                "fill_strategy": "median",
                "fill_value": float(numeric_fill_values[col]),
                "outlier_count": int(clip_bounds[col]["outlier_count"]),
                "clip_lower": float(clip_bounds[col]["lower"]),
                "clip_upper": float(clip_bounds[col]["upper"]),
                "scaled": True,
            }
        )
    for col in categorical_cols:
        summary_rows.append(
            {
                "feature": col,
                "type": "categorical",
                "missing_count": int(missing_counts.get(col, 0)),
                "fill_strategy": "mode",
                "fill_value": str(categorical_fill_values[col]),
                "outlier_count": 0,
                "clip_lower": "",
                "clip_upper": "",
                "scaled": False,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    # Persist all preprocessing rules for inference-time reuse.
    artifacts: dict[str, Any] = {
        "target_col": target_col,
        "drop_cols": drop_cols,
        "raw_feature_columns": X_raw.columns.tolist(),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "numeric_fill_values": numeric_fill_values,
        "categorical_fill_values": categorical_fill_values,
        "clip_bounds": clip_bounds,
        "encoded_feature_columns": X_encoded.columns.tolist(),
        "scaler": scaler,
    }
    return X_scaled, y, summary_df, artifacts


def preprocess_inference_dataframe(
    df: pd.DataFrame,
    artifacts: dict[str, Any],
) -> pd.DataFrame:
    """Apply the saved preprocessing rules to new raw samples."""
    # Apply the stored preprocessing rules to new data.
    working_df = df.copy()
    raw_feature_columns = artifacts["raw_feature_columns"]
    numeric_cols = artifacts["numeric_cols"]
    categorical_cols = artifacts["categorical_cols"]

    # Ensure the same raw feature columns appear in inference data.
    for col in raw_feature_columns:
        if col not in working_df.columns:
            working_df[col] = np.nan

    working_df = working_df[raw_feature_columns]

    # Fill missing values using training medians/modes.
    if numeric_cols:
        working_df[numeric_cols] = working_df[numeric_cols].apply(
            pd.to_numeric, errors="coerce"
        )
        working_df[numeric_cols] = working_df[numeric_cols].fillna(
            value=artifacts["numeric_fill_values"]
        )

    if categorical_cols:
        working_df[categorical_cols] = working_df[categorical_cols].fillna(
            value=artifacts["categorical_fill_values"]
        )

    # Apply the same IQR clipping bounds used during training.
    for col, bounds in artifacts["clip_bounds"].items():
        working_df[col] = working_df[col].clip(
            lower=bounds["lower"],
            upper=bounds["upper"],
        )

    # One-hot encode and align to the training feature order.
    encoded_df = pd.get_dummies(working_df, drop_first=False)
    encoded_df = encoded_df.reindex(
        columns=artifacts["encoded_feature_columns"],
        fill_value=0,
    )

    # Standardize using the training-time scaler.
    scaled_df = pd.DataFrame(
        artifacts["scaler"].transform(encoded_df),
        columns=encoded_df.columns,
        index=encoded_df.index,
    )
    return scaled_df
