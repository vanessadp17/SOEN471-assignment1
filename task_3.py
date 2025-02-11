import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
import graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import export_graphviz

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

# Hyperparameter tuning
params_rand = {
    'n_estimators': [10, 50, 100],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [2, 5, 10],
    'max_features': [0.5, 'sqrt', 'log2'],
}

# Train random forest model using GridSearchCV
print("Training random forest model using GridSearchCV...")
rf_model = RandomForestClassifier(random_state=42)
rand_search = GridSearchCV(rf_model, params_rand, cv=5, scoring="accuracy", n_jobs=-1)
rand_search.fit(x_train, y_train)

# Best parameters
print(f"Best Parameters: {rand_search.best_params_}\n")

# Train the best random forest model
best_rf = rand_search.best_estimator_
# best_rf.fit(pd.DataFrame(x_train, columns=x.columns), y_train)

# Make predictions
y_pred = best_rf.predict(x_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Model performance of Random Forest", end="")
print(f"\nAccuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Display confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="coolwarm", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix of Random Forest Model")
plt.show()

# Define the folder path
output_folder = "random_forest_output_images"
os.makedirs(output_folder, exist_ok=True)

# Visualize the random forest
# Generate the trees
for i in range(5):
    print()
    tree = best_rf.estimators_[i]
    dot_data = export_graphviz(tree, feature_names=x_train.columns, filled=True, max_depth=3, impurity=False, proportion=True)

    filename = os.path.join(output_folder, f"tree_visualization_{i + 1}")
    graph = graphviz.Source(dot_data)
    graph.render(filename, format="png", cleanup=True)

    # Plot the trees
    img = mpimg.imread(f"{filename}.png")
    plt.figure(figsize=(24, 18))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Decision Tree #{i + 1} Visualization")
    plt.show()

    y_pred = tree.predict(x_test.values)
    # Evaluate tree performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"Model performance of Tree #{i + 1}", end="")
    print(f"\nAccuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="coolwarm", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix of Tree #{i + 1} from Random Forest Model")
    plt.show()
