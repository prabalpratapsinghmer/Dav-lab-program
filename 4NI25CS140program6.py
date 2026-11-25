import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

# 1. Load dataset
df = pd.read_csv(r"winequality-red.csv")

print("First five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())

# 2. Separate features and target
X = df.drop("quality", axis=1)
y = df["quality"]

# 3. Normalization (Min-Max Scaling)
normalizer = MinMaxScaler()
X_normalized = normalizer.fit_transform(X)

print("\nNormalized data (first 5 rows):")
print(pd.DataFrame(X_normalized, columns=X.columns).head().round(3))

# 4. Standardization (Z-score Scaling)
standardizer = StandardScaler()
X_standardized = standardizer.fit_transform(X)

print("\nStandardized data (first 5 rows):")
print(pd.DataFrame(X_standardized, columns=X.columns).head().round(3))

# 5. Dimensionality Reduction using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_standardized)

print("\nPCA output (first 5 rows):")
print(pd.DataFrame(X_pca, columns=["PC1", "PC2"]).head().round(3))

print("Original number of features:", X.shape[1])
print("Reduced number of features:", X_pca.shape[1])

print("Variance retained by each component:")
print(pca.explained_variance_ratio_.round(4))

print("Total variance retained:", round(sum(pca.explained_variance_ratio_) * 100, 2), "%")

# 6. Visualize the PCA result
plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Wine Quality Dataset")
plt.colorbar(label="Wine Quality")
plt.show()
