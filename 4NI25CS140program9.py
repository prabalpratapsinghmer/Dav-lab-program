# 1. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# 2. Load dataset
df = pd.read_csv("/home/prabal/Mall_Customers.csv")

print("Dataset shape:", df.shape)
print(df.head())
print(df.describe())
print("Missing values:\n", df.isnull().sum())


# 3. Select features
features = ["Annual Income (k$)", "Spending Score (1-100)"]

# Remove rows with missing feature values
df = df.dropna(subset=features).copy()

X = df[features]


# 4. Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 5. Elbow Method and Silhouette Score
k_values = range(2, 11)

inertias = []
silhouettes = []

for k in k_values:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_scaled)

    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km.labels_))

    print(f"K={k:2d}  Inertia={km.inertia_:8.2f}  Silhouette={silhouettes[-1]:.3f}")


# 6. Plot Elbow and Silhouette graphs
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(list(k_values), inertias, marker="o")
ax[0].set_xlabel("Number of Clusters (K)")
ax[0].set_ylabel("Inertia (WCSS)")
ax[0].set_title("Elbow Method")

ax[1].plot(list(k_values), silhouettes, marker="o", color="green")
ax[1].set_xlabel("Number of Clusters (K)")
ax[1].set_ylabel("Silhouette Score")
ax[1].set_title("Silhouette Method")

plt.tight_layout()
plt.show()


# 7. Apply K-Means Clustering
best_k = 5

kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nCustomers per cluster:")
print(df["Cluster"].value_counts().sort_index())

print("Final Silhouette Score:",
      round(silhouette_score(X_scaled, df["Cluster"]), 3))


# 8. Calculate centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)


# 9. Visualize clusters
plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x=features[0],
    y=features[1],
    hue="Cluster",
    palette="tab10",
    s=70
)

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="X",
    s=250,
    c="black",
    label="Centroids"
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title(f"K-Means Customer Segmentation (K={best_k})")
plt.legend()
plt.tight_layout()
plt.show()
