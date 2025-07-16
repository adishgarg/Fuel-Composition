import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pickle

df = pd.read_csv("synthetic_fuel_data_ctgan.csv")

le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
X = df[['frequency_shift', 'S11_magnitude', 'Q_factor', 'phase_shift']]
y = df['label_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(log_model, f)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)
with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

lgbm_model = LGBMClassifier()
lgbm_model.fit(X_train, y_train)
with open('lightgbm_model.pkl', 'wb') as f:
    pickle.dump(lgbm_model, f)

# Evaluate a sample model (say LightGBM)
y_pred = lgbm_model.predict(X_test)
print("LightGBM Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

y_pred = xgb_model.predict(X_test)
print("XGB Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

y_pred = log_model.predict(X_test)
print("LOG Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

import matplotlib.pyplot as plt
