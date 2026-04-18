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

#Merging the datasets
merged_df = pd.concat([df1, df2], ignore_index=True)

#Print the shape of the merged dataset
print("\nMerged Dataset Shape:", merged_df.shape)

#Saving the merged dataset
merged_df.to_csv("merged_dataset.csv", index=False)

print("\nTask 1 Completed")