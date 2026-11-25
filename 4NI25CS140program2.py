# EDA
import pandas as pd

df = pd.read_csv(r"C:\Users\user\Downloads\archive (1)\student_performance.csv")
print("Dataset loaded successfully!")
print("\nFirst 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())

print("\nShape of the dataset:")
print(df.shape)
print("\nColumn names:")
print(df.columns)
print("\nData types:")
print(df.dtypes)

# Descriptive statistics
print("\nDescriptive statistics:")
print(df.describe())
print("\nStatistics including categorical columns:")
print(df.describe(include="all"))

# Missing values and duplicates
print("\nMissing values in each column:")
print(df.isnull().sum())
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())
print("\nNumber of duplicated student IDs:")
print(df["student_id"].duplicated().sum())

# Grade counts
print("\nNumber of students in each grade:")
print(df["grade"].value_counts())

# Merge dataframes
grade_info = pd.DataFrame({
    "grade": ["A", "B", "C", "D", "F"],
    "grade_description": ["Excellent", "Good", "Average", "Below Average", "Fail"]
})
print("\nGrade information DataFrame:")
print(grade_info)
merged = pd.merge(df, grade_info, on="grade", how="left")
print("\nMerged DataFrame:")
print(merged.head())

# Join dataframe
grade_details = pd.DataFrame({
    "description": ["Excellent", "Good", "Average", "Below Average", "Fail"],
    "Performance level": ["Very High", "High", "Medium", "Low", "Very Low"]
}, index=["A", "B", "C", "D", "F"])
print(grade_details.head())
grade_counts = df["grade"].value_counts().to_frame("student_count")
print(grade_counts.head())

# Concatenate dataframes
df1 = df.iloc[:100]
df2 = df.iloc[100:200]
combined = pd.concat([df1, df2])
print(combined.head())
print(combined.shape)
