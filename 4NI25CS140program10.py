
# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

# 2. Load dataset
df = pd.read_csv("/home/prabal/Mall_Customers.csv")

print(df.shape)
print(df.head())
print("Missing values:", df.isnull().sum())



# 3. Select two features
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# 4. Create dendrogram
Z = linkage(X, method="ward")

plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.show()


# 5. Apply Agglomerative Clustering
model = AgglomerativeClustering(n_clusters=5, linkage="ward")
df["Cluster"] = model.fit_predict(X)

# 6. Display cluster results
print(df[["Annual Income (k$)", "Spending Score (1-100)", "Cluster"]])

# 7. Visualize clusters
plt.figure(figsize=(8, 5))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=df["Cluster"], cmap="rainbow")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation")
plt.show()
