print("Task 01: Merging datasets")
import pandas as pd

#Loading datasets
df1 = pd.read_csv("Credit_Card_Dataset_2025_Sept_1.csv")
df2 = pd.read_csv("Credit_Card_Dataset_2025_Sept_2.csv")

#Checking the data
print("Dataset 1 Preview:")
print(df1.head())

print("\nDataset 2 Preview:")
print(df2.head())

#Checking the structure of the datasets
print("\nShapes:")
print("DF1:", df1.shape)
print("DF2:", df2.shape)

#Checking the columns of the datasets to understand the common key for merging
print("DF1 columns:", df1.columns)
print("DF2 columns:", df2.columns)

#Renaming the common key in dataset 2 to match dataset 1
df2.rename(columns={"User": "ID"}, inplace=True)

#Merging the datasets
merged_df = pd.merge(df1, df2, on="ID", how="inner")

#Print the shape of the merged dataset
print("Merged Shape:", merged_df.shape)

#Saving the merged dataset
merged_df.to_csv("merged_dataset.csv", index=False)