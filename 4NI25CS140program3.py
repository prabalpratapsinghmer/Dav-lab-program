import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load data
df = pd.read_csv(r"D:\kaggle 16-09-26\Iris.csv")

print("First 5 rows of the dataset")
print(df.head())
print("\nLast 5 rows of the dataset")
print(df.tail())
print("\nShape of the dataset")
print(df.shape)
print("\nColumn names")
print(df.columns)
df.info()

# 2. Clean data
print(df.isnull().sum())
df = df.dropna()
print("\n", df.isnull().sum())

print("\n", df.duplicated().sum())
df = df.drop_duplicates()
print("\n", df.duplicated().sum())

print(df.describe())

sns.set_style("whitegrid")

# 3. Histogram
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="SepalLengthCm", bins=10)
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Number of Flowers")
plt.show()

# 4. Boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Species", y="SepalLengthCm")
plt.title("Sepal length by species")
plt.xlabel("Species")
plt.ylabel("Sepal length (cm)")
plt.show()

# 5. Pair plot
sns.pairplot(df, hue="Species")
plt.show()

# 6. Correlation and heatmap
numeric_columns = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
correlation = df[numeric_columns].corr()
print(correlation)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
