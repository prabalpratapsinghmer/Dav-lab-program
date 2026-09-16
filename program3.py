import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df =pd.read_csv(r"D:\kaggle 16-09-26\Iris.csv")

print("First 5 rows of the dataset")
print(df.head())

print("\nLast 5 rows of the dataset")
print(df.tail())

print("\nshape of the dataset")
print(df.shape)

print("\nColumn names")
print(df.columns)

print("\n",df.info())

print(df.isnull().sum())

print("\nremoving the missing values")

df=df.dropna()

print("\n",df.isnull().sum)

#check duplicate rows
print("\n",df.duplicated().sum())

#remove duplicated rows
df=df.drop_duplicates()

print("\n",df.duplicated().sum())

print(df.describe())

sns.set_style("whitegrid")


#creating a histogram
plt.figure(figsize=(8,5))

sns.histplot(data=df, x="SepalLengthCm",bins=10)

plt.title("Distribution of Sepal Length")

plt.xlabel("Sepal Length (cm)")

plt.ylabel("Number of Flowers")

plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(data=df,x="Species",y="SepalLengthCm")
plt.title("Sepal length by species")
plt.xlabel("species")
plt.ylabel("Sepal length (cm)")
plt.show()




#pair plot
sns.pairplot(df,hue="Species")

plt.show()

#correlation

numeric_columns=[
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]


correlation=df[numeric_columns].corr()

print(correlation)

#heatmap
plt.figure(figsize=(8,6))
sns.heatmap(
    correlation,
    annot=True,
    Cmap="coolwarm",
    fmt=".2f"
)

plt.title("correlation heatmap")
plt.show()