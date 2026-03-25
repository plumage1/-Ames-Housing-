import joblib
import matplotlib.pyplot as plt
import pandas as pd

from preprocessing_utils import preprocess_inference_dataframe
from project_paths import OUTPUT_DIR, RAW_DATA_DIR, RESULT_DIR


# Ensure output folders exist for prediction artifacts.
OUTPUT_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

print("=" * 50)
print("House price prediction")
print("=" * 50)

model_path = OUTPUT_DIR / "best_model.pkl"
artifacts_path = OUTPUT_DIR / "preprocessing_artifacts.pkl"
best_model_name_path = OUTPUT_DIR / "best_model_name.txt"

if not model_path.exists():
    raise FileNotFoundError("No trained model found. Please run code/04_model_training.py first.")
if not artifacts_path.exists():
    raise FileNotFoundError("No preprocessing artifacts found. Please run code/03_data_preprocessing.py first.")

model = joblib.load(model_path)
artifacts = joblib.load(artifacts_path)
best_model_name = best_model_name_path.read_text(encoding="utf-8").strip() if best_model_name_path.exists() else "best_model"

print("Loaded model:", best_model_name)
print("Feature count:", len(artifacts["encoded_feature_columns"]))

# This sample mimics a single new property record in the raw schema.
new_house = {
    "MSSubClass": 60,
    "LotFrontage": 70.0,
    "LotArea": 8450,
    "OverallQual": 7,
    "OverallCond": 5,
    "YearBuilt": 2000,
    "YearRemodAdd": 2000,
    "MasVnrArea": 0.0,
    "BsmtFinSF1": 500,
    "BsmtFinSF2": 0.0,
    "BsmtUnfSF": 400,
    "TotalBsmtSF": 900,
    "1stFlrSF": 1000,
    "2ndFlrSF": 500,
    "GrLivArea": 1500,
    "BsmtFullBath": 1,
    "BsmtHalfBath": 0,
    "FullBath": 2,
    "HalfBath": 1,
    "BedroomAbvGr": 3,
    "KitchenAbvGr": 1,
    "TotRmsAbvGrd": 6,
    "Fireplaces": 1,
    "GarageYrBlt": 2000,
    "GarageCars": 2,
    "GarageArea": 400,
    "WoodDeckSF": 100,
    "OpenPorchSF": 50,
    "EnclosedPorch": 0,
    "3SsnPorch": 0,
    "ScreenPorch": 0,
    "PoolArea": 0,
    "MiscVal": 0,
    "MoSold": 6,
    "YrSold": 2010,
    "MSZoning": "RL",
    "Neighborhood": "NAmes",
    "HouseStyle": "2Story",
    "KitchenQual": "Gd",
    "GarageType": "Attchd",
}

# Apply the training-time preprocessing rules to new input.
raw_input_df = pd.DataFrame([new_house])
processed_input_df = preprocess_inference_dataframe(raw_input_df, artifacts)
prediction = model.predict(processed_input_df)[0]

training_saleprice = pd.read_csv(RAW_DATA_DIR / "train.csv")["SalePrice"]
training_mean_price = training_saleprice.mean()
training_median_price = training_saleprice.median()

print("Predicted sale price: ${:,.2f}".format(prediction))
print("Predicted sale price (x10k): {:.2f}".format(prediction / 10000))

# Save the raw and processed inputs for reproducibility.
raw_input_df.to_csv(OUTPUT_DIR / "prediction_input_raw.csv", index=False)
processed_input_df.to_csv(OUTPUT_DIR / "prediction_input_processed.csv", index=False)

result_df = pd.DataFrame(
    {
        "model_name": [best_model_name],
        "predicted_sale_price": [prediction],
        "predicted_sale_price_10k": [prediction / 10000],
        "training_mean_price": [training_mean_price],
        "training_median_price": [training_median_price],
    }
)
result_df.to_csv(OUTPUT_DIR / "prediction_result.csv", index=False)

# Visualize the prediction against training benchmarks.
fig, ax = plt.subplots(figsize=(7, 4))
labels = ["Predicted", "Training Mean", "Training Median"]
values = [prediction, training_mean_price, training_median_price]
colors = ["#1b9e77", "#d95f02", "#7570b3"]
ax.bar(labels, values, color=colors)
ax.set_title("Prediction vs Training Price Benchmarks")
ax.set_ylabel("Price ($)")
for idx, value in enumerate(values):
    ax.text(idx, value, f"${value:,.0f}", ha="center", va="bottom")
fig.tight_layout()
fig.savefig(RESULT_DIR / "prediction_visualization.png", dpi=300)
plt.close(fig)

print("Saved raw prediction input to:", OUTPUT_DIR / "prediction_input_raw.csv")
print("Saved processed prediction input to:", OUTPUT_DIR / "prediction_input_processed.csv")
print("Saved prediction result to:", OUTPUT_DIR / "prediction_result.csv")
print("Saved prediction visualization to:", RESULT_DIR / "prediction_visualization.png")
print("=" * 50)
print("Prediction completed.")
