import matplotlib.pyplot as plt
import pandas as pd

from project_paths import RAW_DATA_DIR, RESULT_DIR


DATA_PATH = RAW_DATA_DIR / "train.csv"
RESULT_DIR.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# Visualize the raw training data before preprocessing.
df = pd.read_csv(DATA_PATH)

print("=" * 50)
print("Creating data exploration charts...")
print("=" * 50)

# Create a 2x2 canvas to compare multiple distributions side by side.
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Histogram of sale prices.
axes[0, 0].hist(df["SalePrice"], bins=50, color="skyblue", edgecolor="black")
axes[0, 0].set_title("SalePrice Distribution")
axes[0, 0].set_xlabel("Price ($)")
axes[0, 0].set_ylabel("Count")

# Boxplot highlights median, quartiles, and outliers.
axes[0, 1].boxplot(df["SalePrice"], vert=True, patch_artist=True)
axes[0, 1].set_title("SalePrice Boxplot")
axes[0, 1].set_ylabel("Price ($)")

# Scatter plot of living area vs price.
axes[1, 0].scatter(df["GrLivArea"], df["SalePrice"], alpha=0.5, color="green")
axes[1, 0].set_title("GrLivArea vs SalePrice")
axes[1, 0].set_xlabel("GrLivArea")
axes[1, 0].set_ylabel("Price ($)")

# Compare price distributions by bedroom count.
bedroom_groups = df.groupby("BedroomAbvGr")["SalePrice"].apply(list)
axes[1, 1].boxplot(
    [bedroom_groups[label] for label in bedroom_groups.index],
    tick_labels=list(bedroom_groups.index),
    patch_artist=True,
)
axes[1, 1].set_title("BedroomAbvGr vs SalePrice")
axes[1, 1].set_xlabel("BedroomAbvGr")
axes[1, 1].set_ylabel("Price ($)")

plt.tight_layout()
# Save the visualization for reporting.
save_path = RESULT_DIR / "01_data_visualization.png"
plt.savefig(save_path, dpi=300)
plt.close(fig)

print(f"Saved chart to: {save_path}")
print("=" * 50)
print("Visualization completed.")
