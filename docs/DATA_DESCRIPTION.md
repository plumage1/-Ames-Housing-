# Data Description

## Dataset source

The raw dataset in this project corresponds to the Ames Housing dataset from the Kaggle competition:

- Kaggle: House Prices - Advanced Regression Techniques
- Source link: `https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data`

## Raw data files

The original files are stored in `data/raw/`:

- `train.csv`: training dataset with target column `SalePrice`
- `test.csv`: test dataset without the target column

## Target variable

- `SalePrice`: house selling price

## Example feature fields

Representative fields in the raw dataset include:

- `OverallQual`: overall material and finish quality
- `GrLivArea`: above-grade living area
- `GarageCars`: garage capacity in number of cars
- `GarageArea`: garage size in square feet
- `YearBuilt`: original construction year
- `Neighborhood`: neighborhood category
- `KitchenQual`: kitchen quality

## Preprocessing steps

The preprocessing pipeline implemented in `code/03_data_preprocessing.py` includes:

1. Remove the identifier column `Id`
2. Split features and target `SalePrice`
3. Fill missing numeric values with the median
4. Fill missing categorical values with the mode
5. Detect and clip numeric outliers using the IQR rule
6. Apply one-hot encoding to categorical variables and keep the full category set
7. Standardize the final feature matrix using `StandardScaler`
8. Save preprocessing artifacts for consistent inference

## Processed data files

The processed files are stored in `data/processed/`:

- `X_train.csv`: processed training feature matrix
- `y_train.csv`: training target values
- `train_processed.csv`: combined processed training dataset
- `X_test_processed.csv`: processed test feature matrix

## Supporting files

- `docs/preprocessing_summary.csv`: summary of missing-value handling, outlier processing, and scaling
- `output/preprocessing_artifacts.pkl`: reusable preprocessing artifacts
- `output/feature_columns.csv`: saved training feature order for inference alignment
