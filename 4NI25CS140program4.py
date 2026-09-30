import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r"D:\archive\melb_data.csv")

print("Original dataset:")
print(df.head())

print("\nShape of dataset:")
print(df.shape)

print("\nData information:")
print(df.info())

print("\nColumn names:")
print(df.columns.tolist())

print("\nStatistical summary:")
print(df.describe())

print("\nMissing values before preprocessing:")
print(df.isnull().sum())

df_mean = df.copy()

mean_building_area = df_mean["BuildingArea"].mean()

df_mean["BuildingArea"] = df_mean["BuildingArea"].fillna(
    mean_building_area
)

print("\nMean BuildingArea:", mean_building_area)

print("\nMissing BuildingArea values after mean imputation:")
print(df_mean["BuildingArea"].isnull().sum())

df_median = df.copy()

median_year_built = df_median["YearBuilt"].median()

df_median["YearBuilt"] = df_median["YearBuilt"].fillna(
    median_year_built
)

print("\nMedian YearBuilt:", median_year_built)

print("\nMissing YearBuilt values after median imputation:")
print(df_median["YearBuilt"].isnull().sum())

df_ffill = df.copy()

df_ffill["Car"] = df_ffill["Car"].ffill()

print("\nMissing Car values after forward fill:")
print(df_ffill["Car"].isnull().sum())

comparison = pd.DataFrame({
    "Original_BuildingArea": df["BuildingArea"].head(15),
    "Mean_Imputed_BuildingArea": df_mean["BuildingArea"].head(15),
    "Original_YearBuilt": df["YearBuilt"].head(15),
    "Median_Imputed_YearBuilt": df_median["YearBuilt"].head(15),
    "Original_Car": df["Car"].head(15),
    "Forward_Filled_Car": df_ffill["Car"].head(15)
})

print("\nComparison of imputation methods:")
print(comparison)

df_processed = df.copy()

council_mode = df_processed["CouncilArea"].mode()[0]

df_processed["CouncilArea"] = df_processed["CouncilArea"].fillna(
    council_mode
)

print("\nCouncilArea mode:", council_mode)

print("\nMissing CouncilArea values after mode imputation:")
print(df_processed["CouncilArea"].isnull().sum())

df_encoded = pd.get_dummies(
    df_processed,
    columns=["Type", "Method"],
    dtype=bool
)

print("\nDataset after one-hot encoding:")
print(df_encoded.head())

df_label = df_processed.copy()

label_encoder = LabelEncoder()

df_label["CouncilArea_encoded"] = label_encoder.fit_transform(
    df_label["CouncilArea"]
)

print("\nLabel encoded CouncilArea:")

print(
    df_label[
        ["CouncilArea", "CouncilArea_encoded"]
    ].head(10)
)

print("\nMissing values after preprocessing:")
print(df_processed.isnull().sum())