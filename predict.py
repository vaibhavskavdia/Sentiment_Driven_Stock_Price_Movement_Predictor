import pandas as pd
import joblib
import os
# Load the saved model
model = joblib.load("models/adaboost_stock_predictor.pkl")

# Load new data (you must have already preprocessed it with the same feature pipeline)
df = pd.read_csv("model_data/new_data.csv")

# Use the same feature columns as training
features = ["Open", "High", "Low", "Close", "Volume", "sentiment_score",
            "price_change", "volatility", "hour", "minute"]

X_new = df[features]

# Predict
df["predicted_movement"] = model.predict(X_new)
os.makedirs("predictions",exist_ok=True)
# Save or view predictions
df.to_csv("predictions/new_predictions.csv", index=False)
print("✅ Predictions saved to model_data/new_predictions.csv")
