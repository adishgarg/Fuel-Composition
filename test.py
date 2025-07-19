import pickle
import numpy as np
import pandas as pd


with open('lightgbm_model.pkl', 'rb') as f:
    lgbm_model = pickle.load(f)

with open('xgboost_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

with open('logistic_regression_model.pkl', 'rb') as f:
    log_model = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Test sample: [frequency_shift, S11_magnitude, Q_factor, phase_shift]
test_sample = np.array([
    [25, 1.65, 0.74, -3.36, 9.43],   
])

lgbm_pred = le.inverse_transform(lgbm_model.predict(test_sample).astype(int))
xgb_pred  = le.inverse_transform(xgb_model.predict(test_sample).astype(int))
log_pred  = le.inverse_transform(log_model.predict(test_sample).astype(int))

for idx, sample in enumerate(test_sample):
    print(f"\nSample {idx+1} : {sample}")
    # print(f"  🔸 LightGBM Prediction: {lgbm_pred[idx]}")
    print(f"  🔸 XGBoost Prediction : {xgb_pred[idx]}")
    # print(f"  🔸 Logistic Regression: {log_pred[idx]}")