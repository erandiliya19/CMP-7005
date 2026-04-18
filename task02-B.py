import pandas as pd

#Load the merged dataset
df = pd.read_csv("merged_dataset.csv")

#Part B - Data pre processing 

import pandas as pd

#Load the merged dataset
df = pd.read_csv("merged_dataset.csv")

#Clean the column names 
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

#removing duplicates from the dataset
print("Before:", df.shape)
df.drop_duplicates(inplace=True)
print("After:", df.shape)
print("Duplicate rows:", df.duplicated().sum())

#Missing Value handling 
print("Before cleaning:")
print(df.isnull().sum())

#Numerical columns
df.fillna(df.median(numeric_only=True), inplace=True)

#Categorical columns
df.fillna("unknown", inplace=True)

print("After cleaning:")
print(df.isnull().sum())

#Handle Outliers
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=df['income'])
plt.title("Income Outliers")
plt.show()

Q1 = df['income'].quantile(0.25)
Q3 = df['income'].quantile(0.75)
IQR = Q3 - Q1

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)

df = df[(df['income'] >= Q1 - 1.5 * IQR) & 
        (df['income'] <= Q3 + 1.5 * IQR)]

print("After removing outliers:")
print(df['income'].describe())



