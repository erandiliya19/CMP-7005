import pandas as pd

#Load the merged dataset
df = pd.read_csv("merged_dataset.csv")

#Part B - Data pre processing 

#Clean the column names 
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

#removing duplicates from the dataset
print("Before:", df.shape)
df.drop_duplicates(inplace=True)
print("After:", df.shape)
#Verification of duplicates
print("Duplicate rows:", df.duplicated().sum())

#Missing value handling 
print("Before cleaning:")
print(df.isnull().sum())
#Numerical columns - Median method
df.fillna(df.median(numeric_only=True), inplace=True)
#Categorical columns - Unknown method
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

#Remove unrealistic ages
df = df[(df['age'] >= 18) & (df['age'] <= 100)]

#Feature Engineering 
#Age Group 
df['age_group'] = pd.cut(df['age'],
                         bins=[18, 30, 45, 60, 100],
                         labels=['young', 'adult', 'middle_aged', 'senior'])
#Employment category
df['employment_level'] = pd.cut(df['years_employed'],
                                bins=[0, 2, 5, 10, 50],
                                labels=['new', 'junior', 'mid', 'experienced'])
#Income level
df['income_level'] = pd.cut(df['income'],
                            bins=3,
                            labels=['low', 'medium', 'high'])

#Drop irrelevant columns
df.drop(columns=['unnamed:_0', 'id', 'flag_mobil'], inplace=True)

#Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

#Updated dataset overview
print(df.info())
print(df.isnull().sum())
print(df.head())

print("Final Shape:", df.shape)

#Saving the updated dataset
df.to_csv("updated_dataset.csv", index=False)
print("Dataset saved successfully")