# eval_performance.py
import joblib
from sklearn.metrics import mean_absolute_error
import pandas as pd

# Load model
model_pipeline = joblib.load("model_pipeline.joblib")

# Load test data
test_data = pd.read_csv("data/data_test.csv")
test_data = test_data.drop(columns=[test_data.columns[0]])  # remove id column

# Prepare test features and target
TARGET = "Final_CGPA"
X_test = test_data.drop(columns=[TARGET])
y_test = test_data[TARGET]

# Predict on test data
y_pred = model_pipeline.predict(X_test)
# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error on test set: {mae:.4f}")
