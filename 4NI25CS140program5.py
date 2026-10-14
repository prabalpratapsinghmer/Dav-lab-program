import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Kamal Kant Singh\Desktop\archive\StudentsPerformance.csv")

print("first 5 rows:")
print(df.head())

print("\ndata set shape:")
print(df.shape)

data = df["math score"].dropna()
print("\nnumber of math score values:", len(data))

# IQR method
q1 = data.quantile(0.25)
q3 = data.quantile(0.75)
iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

print("\niqr method")
print("q1:", q1)
print("q3:", q3)
print("iqr:", iqr)
print("lower limit:", lower)
print("upper limit:", upper)

iqr_outliers = df[
    (df["math score"] < lower) |
    (df["math score"] > upper)
]

print("number of iqr outliers:", len(iqr_outliers))

# Z-score method
mean = data.mean()
std = data.std()

df["z_score"] = (df["math score"] - mean) / std  # FIXED: use underscore consistently
z_outliers = df[abs(df["z_score"]) > 3]

print("\nz method")
print("mean", mean)
print("standard deviation", std)
print("number of z outliers", len(z_outliers))

# Print outliers
print("\niqr outliers")
if len(iqr_outliers) > 0:
    print(iqr_outliers[["math score"]])
else:
    print("no iqr outliers found")

print("\nz score outliers")
if len(z_outliers) > 0:
    print(z_outliers[["math score", "z_score"]])  # FIXED: column name
else:
    print("no z score outliers found")

# Treatment: capping using IQR limits
df["math_score_treated"] = df["math score"].clip(lower=lower, upper=upper)

print("\ntreatment method: capping using iqr limits:")

print("\nbefore treatment")
print(df["math score"].describe())

print("\nafter treatment")
print(df["math_score_treated"].describe())

comparison = df[["math score", "math_score_treated"]]
print(comparison.head(20))

# Visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.boxplot(df["math score"].dropna())
plt.title("math score - before treatment")
plt.ylabel("math score")

plt.subplot(1, 2, 2)
plt.boxplot(df["math_score_treated"].dropna())
plt.title("math score - after treatment")
plt.ylabel("math score")

plt.tight_layout()
plt.show()