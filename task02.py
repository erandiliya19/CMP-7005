import pandas as pd

#Load the merged dataset
df = pd.read_csv("merged_dataset.csv")

#Part A - Understanding the merged dataset

#Number of rows and colums in the merged dataset 
print("Number of Rows and Columns:")
print(df.shape)

#Unique values in data
print("\nUnique values in each column:")
print(df.nunique())

#Data types of each column 
print("\nData Types:")
print(df.info())

#Distribution of data - Numerical 
print("\nStatistical Summary:")
print(df.describe())

#Disctribution of data - Target 
print("\nTarget Variable Distribution:")
print(df['TARGET'].value_counts())

#Missing values in the dataset
print("\nMissing Values in Dataset:")
print(df.isnull().sum())



