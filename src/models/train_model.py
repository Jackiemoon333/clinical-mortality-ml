import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Project root
csv_path = BASE_DIR / "data" / "processed" / "mortality_features.csv"

# Load CSV
df = pd.read_csv(csv_path)
print("Initial data:")
print(df.head())

# -----------------------------
# Clean age column
# -----------------------------
df['age'] = df['age'].replace('> 89', 90)           # Replace '> 89' with 90
df['age'] = pd.to_numeric(df['age'], errors='coerce')  # Convert to numeric
df['age'] = df['age'].fillna(df['age'].median())       # Fill NaNs

# -----------------------------
# Define target and features
# -----------------------------
y = df["target_mortality"]
X = df.drop(columns=["target_mortality", "patientunitstayid"])

# -----------------------------
# Categorical and numeric columns
# -----------------------------
cat_cols = ["gender", "ethnicity"]
num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

# -----------------------------
# Preprocessing
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", SimpleImputer(strategy="median"), num_cols)
    ]
)

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Logistic Regression
# -----------------------------
log_model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000))
    ]
)

log_model.fit(X_train, y_train)
log_preds = log_model.predict_proba(X_test)[:, 1]
print("Logistic Regression ROC-AUC:", roc_auc_score(y_test, log_preds))

# -----------------------------
# Random Forest
# -----------------------------
rf_model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(class_weight="balanced", n_estimators=200))
    ]
)

rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict_proba(X_test)[:, 1]
print("Random Forest ROC-AUC:", roc_auc_score(y_test, rf_preds))

# 5-fold cross-validation
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring="roc_auc")
cv_mean = float(cv_scores.mean())
cv_std = float(cv_scores.std())
print(f"5-fold CV ROC-AUC: {cv_mean:.3f} (+/- {cv_std:.3f})")

# -----------------------------
# Feature Importance
# -----------------------------

import numpy as np

# Get categorical feature names after OneHotEncoding
ohe = rf_model.named_steps["preprocess"].named_transformers_["cat"]
cat_feature_names = ohe.get_feature_names_out(cat_cols)

# Get numeric column names automatically
num_cols = [col for col in X.columns if col not in cat_cols]

# Combine them
feature_names = list(cat_feature_names) + num_cols

# Get feature importance from Random Forest
importances = rf_model.named_steps["model"].feature_importances_

# If lengths mismatch, trim to match
feature_names = feature_names[:len(importances)]

# Create dataframe
feat_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

# Sort
feat_importance = feat_importance.sort_values(by="importance", ascending=False)

print("\nTop 10 Important Features:")
print(feat_importance.head(10))

#import matplotlib.pyplot as plt

#top_features = feat_importance.head(10)

#plt.figure(figsize=(8,5))
#plt.barh(top_features["feature"], top_features["importance"])
#plt.gca().invert_yaxis()
#plt.title("Top Mortality Prediction Features")
#plt.xlabel("Feature Importance")
#plt.tight_layout()

#plt.show()

import joblib
import json
from pathlib import Path

# create models folder if it doesn't exist
model_dir = BASE_DIR / "models"
model_dir.mkdir(exist_ok=True)

model_path = model_dir / "mortality_model.pkl"
joblib.dump(rf_model, model_path)
print(f"Model saved to {model_path}")

log_model_path = model_dir / "logistic_model.pkl"
joblib.dump(log_model, log_model_path)
print(f"Logistic model saved to {log_model_path}")

# Save metrics for app display
rf_auc = roc_auc_score(y_test, rf_preds)
log_auc = roc_auc_score(y_test, log_preds)
metrics = {
    "roc_auc": float(rf_auc),
    "cv_roc_auc_mean": cv_mean,
    "cv_roc_auc_std": cv_std,
    "logistic_roc_auc": float(log_auc),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "n_features": int(X_train.shape[1]),
    "class_balance": {
        "train_pos": int(y_train.sum()),
        "train_neg": int((1 - y_train).sum()),
        "test_pos": int(y_test.sum()),
        "test_neg": int((1 - y_test).sum()),
    },
    "model_type": "RandomForest",
    "n_estimators": 200,
}
with open(model_dir / "model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved to {model_dir / 'model_metrics.json'}")