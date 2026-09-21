#introduction to python for data analytics suing numpy , pandas , basic array operation , indexing , slicing and dataframe creation
import numpy as np
import pandas as pd
arr=np.array([10,20,30,40,50])
print("array",arr)
print("mean",arr.mean())
print("reshaped array :",arr.reshape(5,1))
print("sliced index(1 to 3):",arr[1:4])
data={"name":["asha","ravi","meera","karan"],
       "mark":[88,72,12,34],
       "branch":["CSE","ECE","mech","EEE"]}
df=pd.DataFrame(data)
print(df)
print("first two rows :\n",df.iloc[0:2])
print("\nmarks column:\n",df["mark"])
print("\n rows where marks>70 :\n ",df[df["mark"]>70])