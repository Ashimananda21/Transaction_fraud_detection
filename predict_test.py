import joblib
import pandas as pd
from src.preprocess import clean_data, engineer_features, prepare_dataset

# Load trained model
model = joblib.load("models/best_model.joblib")

# Sample transaction
sample = pd.DataFrame([{
    "transaction_id": "T99999",
    "user_id": "U99999",
    "timestamp": "2026-04-02 22:50:00",
    "amount": 95000,
    "merchant": "LuxuryStore",
    "category": "electronics",
    "device": "mobile",
    "ip_address": "192.168.1.100",
    "is_fraud": 0
}])

# Same preprocessing as training
sample = clean_data(sample)
sample = engineer_features(sample)
sample_X, _ = prepare_dataset(sample)

# Remove ID-like columns like training
drop_cols = [col for col in sample_X.columns if "id" in col.lower()]
sample_X = sample_X.drop(columns=drop_cols, errors="ignore")

# Drop constant columns
sample_X = sample_X.loc[:, sample_X.apply(pd.Series.nunique) > 0]

# Align sample columns with model training columns
expected_cols = model.named_steps["pre"].feature_names_in_

for col in expected_cols:
    if col not in sample_X.columns:
        sample_X[col] = 0

sample_X = sample_X[expected_cols]

# Predict
prediction = model.predict(sample_X)[0]
probability = model.predict_proba(sample_X)[0][1]

print("Prediction:", "Fraud" if prediction == 1 else "Not Fraud")
print("Fraud Probability:", round(probability, 4))