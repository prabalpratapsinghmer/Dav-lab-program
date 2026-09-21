#lab 2 explainatory data analysis 
import pandas as pd
df=pd.read_csv(r"C:\Users\user\Downloads\archive (1)\student_performance.csv")
print("dataset loaded successfully!")
print("\nfirst 5 rows:");
print(df.head())
print("\nlast 5 rows:");
print(df.tail());


print("\nShape of the dataset:")
print(df.shape)
print("\nColumn names:")
print(df.columns)
print("\nData types:")
print(df.dtypes)




#EDA exploratory data analysis


print("\n Descriptive statistics:")
print(df.describe())




print("\n statistics including catagorical columns:")
print(df.describe(include="all"))


#check missing value
print("\nchecking missing values in each column")
print(df.isnull().sum())


#checking for duplicate rows
print("\n number of duplicate rows")
print(df.duplicated().sum())



print("\n number of duplicated student ID's")
print(df['student_id'].duplicated().sum())


#display grade counts
print("\n number of students in each grade")
print(df['grade'].value_counts())


#data wrangling create a new dataset for merging
grade_info=pd.DataFrame({
    'grade':['A','B','C','D','F'],
    'grade_description':[
        'Excellent',
        'Good',
        'Average',
        'Below Average',
        'Fail'
    ]
})
print("\n Grade information Dataframe:")
print(grade_info)
merged=pd.merge(
    df,
    grade_info,
    on='grade',
    how='left'
)
print("\nMerged DataFrame:")
print(merged.head())



#create dataframe for join
grade_details=pd.DataFrame({
    'description':[
        'Excellent',
        'Good',
        'Average',
        'Below Average',
        'Fail'
    ],
    'Performance level':[
        'Very High',
        'High',
        'Medium',
        'Low',
        'Very Low'
    ]
}, index=['A','B','C','D','F'])
print(grade_details.head())
grade_counts=df['grade'].value_counts().to_frame('student_count')
print(grade_counts.head())



#split datafram into two parts
df1=df.iloc[:100]
df2=df.iloc[100:200]
combined=pd.concat([df1,df2])
print(combined.head())
print(combined.shape)