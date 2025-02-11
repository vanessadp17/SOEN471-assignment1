import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "data/customer_churn.csv"
df = pd.read_csv(file_path)

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().any())

# Drop rows with missing data
df = df.dropna()

# Ensure no null values remain
print("\nEnsure No Null Values after dropna:")
print(df.isnull().any())

# Visualize data distributions using Histogram
plt.figure(figsize=(12, 8))
df.hist(figsize=(12, 10), bins=20, edgecolor='black')
plt.suptitle("Distribution of Numerical Features", fontsize=14)
plt.show()

# Visualize data distributions using Box plots
plt.figure(figsize=(12, 8))
sns.boxplot(data=df.drop(columns=["CustomerID", "Churn"]))
plt.xticks(rotation=45)
plt.title("Box Plot of Numerical Features")
plt.show()

# Correlation matrix heatmap between variables
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()






