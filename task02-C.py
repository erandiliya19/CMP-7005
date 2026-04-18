#Import dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("updated_dataset.csv")

#Univariate analysis - Single variable
#Target
sns.countplot(x=df['target'])
plt.title("Target Distribution (Fraud vs Non-Fraud)")
plt.savefig("target.png")
plt.show()

#Income distribution 
sns.histplot(df['income'], bins=30, kde=True)
plt.title("Income Distribution")
plt.savefig("income.png")
plt.show()

#Age distribution
sns.histplot(df['age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.savefig("age.png")
plt.show()

#Bivariate analysis - Two variables
#Income vs Target
sns.boxplot(x=df['target'], y=df['income'])
plt.title("Income vs Fraud")
plt.savefig("income_vs_target.png")
plt.show()

#Age vs Target
sns.boxplot(x=df['target'], y=df['age'])
plt.title("Age vs Fraud")
plt.savefig("age_vs_target.png")
plt.show()

#Multivariate analysis 
#Income level vs Target
sns.countplot(x='income_level', hue='target', data=df)
plt.title("Income Level vs Fraud")
plt.savefig("income_level_vs_target.png")
plt.show()

#Emplyment level vs Target
sns.countplot(x='employment_level', hue='target', data=df)
plt.title("Employment Level vs Fraud")
plt.savefig("employment_level_vs_target.png")
plt.show()

#Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

#Statistical summary
print(df.describe())