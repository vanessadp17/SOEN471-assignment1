import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Load dataset
file_path = "data/customer_churn.csv"
df = pd.read_csv(file_path)

# Drop missing values
df = df.dropna()

# Categorical variables to numerical
df = pd.get_dummies(df, drop_first=True)

x = df.drop(columns=["Churn"])  # Assuming 'Churn' is the target variable
y = df["Churn"]

# Split dataset into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Hyperparameters grid for GridSearchCV
param_grid = {
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"]
}

# Train decision tree using GridSearchCV
dt_model = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dt_model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(x_train, y_train)

# Best parameters
print(f"Best Parameters: {grid_search.best_params_}")

# Train the best decision tree model
best_dt = grid_search.best_estimator_
best_dt.fit(x_train, y_train)

# Make predictions
y_pred = best_dt.predict(x_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="coolwarm", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix of Decision Tree Model")
plt.show()

# Visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(best_dt, feature_names=x.columns, class_names=["No Churn", "Churn"], filled=True, rounded=True, fontsize=8)
plt.title("Decision Tree Visualization")
plt.show()
