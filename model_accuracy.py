import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

with open('lightgbm_model.pkl', 'rb') as f:
    lgbm_model = pickle.load(f)

with open('xgboost_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

with open('logistic_regression_model.pkl', 'rb') as f:
    log_model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

df = pd.read_csv("synthetic_fuel_data_gc.csv")  
df['label_encoded'] = le.transform(df['label'])
X = df[['frequency_shift', 'S11_magnitude', 'Q_factor', 'phase_shift']]
y_true = df['label_encoded']

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

lgbm_pred = lgbm_model.predict(X)
xgb_pred = xgb_model.predict(X)
log_pred = log_model.predict(X)

acc_lgbm = accuracy_score(y_true, lgbm_pred)
acc_xgb = accuracy_score(y_true, xgb_pred)
acc_log = accuracy_score(y_true, log_pred)

plt.figure(figsize=(6, 4))
models = ['LightGBM', 'XGBoost', 'Logistic Regression']
accuracies = [acc_lgbm, acc_xgb, acc_log]
sns.barplot(x=models, y=accuracies, palette='viridis')
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy Score")
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

def plot_confusion(model_name, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

plot_confusion("LightGBM", lgbm_pred)
plot_confusion("XGBoost", xgb_pred)
plot_confusion("Logistic Regression", log_pred)

def classification_report_to_df(y_true, y_pred, model_name):
    report = classification_report(y_true, y_pred, output_dict=True, target_names=le.classes_)
    df_report = pd.DataFrame(report).transpose().drop(index=['accuracy', 'macro avg', 'weighted avg'])
    sns.heatmap(df_report.iloc[:, :-1], annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title(f"{model_name} Classification Report")
    plt.tight_layout()
    plt.show()

classification_report_to_df(y_true, lgbm_pred, "LightGBM")
classification_report_to_df(y_true, xgb_pred, "XGBoost")
classification_report_to_df(y_true, log_pred, "Logistic Regression")