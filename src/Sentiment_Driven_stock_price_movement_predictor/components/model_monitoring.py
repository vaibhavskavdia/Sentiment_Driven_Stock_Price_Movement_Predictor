import joblib
import os
from sklearn.ensemble import AdaBoostRegressor
import pandas as pd

# Load preprocessed data
X = pd.read_csv("notebook/model_data/X.csv")
y = pd.read_csv("notebook/model_data/y.csv")

# Train model
model = AdaBoostRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Ensure output directory exists
os.makedirs("models", exist_ok=True)

# Save model to file
joblib.dump(model, "models/adaboost_stock_predictor.pkl")
print("✅ Model saved successfully to models/adaboost_stock_predictor.pkl")
