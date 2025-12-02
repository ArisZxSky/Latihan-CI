import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Tidak perlu set_tracking_uri di CI/CD
mlflow.set_experiment("Latihan Credit Scoring")

# Enable auto logging
mlflow.autolog()

# Load dataset
data = pd.read_csv("train_pca.csv")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data.drop("Credit_Score", axis=1),
    data["Credit_Score"],
    random_state=42,
    test_size=0.2
)

# Example for model signature
input_example = X_train[0:5]

# Parameter model
n_estimators = 505
max_depth = 37

# Log params (optional, auto-log also logs them)
mlflow.log_param("n_estimators", n_estimators)
mlflow.log_param("max_depth", max_depth)

# Train model
model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth
)

model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)

# Log metric
mlflow.log_metric("accuracy", accuracy)

# Log model explicitly with input example
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    input_example=input_example
)

print("Training completed successfully with accuracy:", accuracy)
