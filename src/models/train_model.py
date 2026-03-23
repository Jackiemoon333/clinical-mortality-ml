import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
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
# Random Forest with hyperparameter tuning
# -----------------------------
# Param grid: n_estimators, max_depth, min_samples_*, max_features
param_dist = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [8, 12, 16, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2"],
}
rf_base = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(class_weight="balanced", random_state=42))
    ]
)
search = RandomizedSearchCV(
    rf_base,
    param_distributions=param_dist,
    n_iter=24,  # 24 random combinations; 5-fold CV = 120 fits
    cv=5,
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
print("Running hyperparameter search (24 configs x 5-fold CV)...")
search.fit(X_train, y_train)
rf_model = search.best_estimator_
print(f"Best params: {search.best_params_}")
print(f"Best 5-fold CV ROC-AUC: {search.best_score_:.3f}")

rf_preds = rf_model.predict_proba(X_test)[:, 1]
print("Random Forest (tuned) test ROC-AUC:", roc_auc_score(y_test, rf_preds))

cv_mean = float(search.best_score_)
cv_std = float(search.cv_results_["std_test_score"][search.best_index_])

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

# -----------------------------
# Fairness: stratified ROC-AUC by age, gender, ethnicity
# -----------------------------
def compute_fairness_metrics(X_test_raw, y_test, preds):
    """Compute ROC-AUC per subgroup. Returns list of dicts for JSON."""
    results = []
    X_test_raw = X_test_raw.copy()
    # Clean age for grouping
    X_test_raw["age_clean"] = X_test_raw["age"].replace("> 89", 90)
    X_test_raw["age_clean"] = pd.to_numeric(X_test_raw["age_clean"], errors="coerce")
    X_test_raw["age_clean"] = X_test_raw["age_clean"].fillna(X_test_raw["age_clean"].median())

    def add_subgroup(name, mask):
        n = int(mask.sum())
        m = mask.values if hasattr(mask, "values") else mask
        n_pos = int(np.sum(np.asarray(y_test)[m]))
        if n < 20 or n_pos < 3:  # ROC-AUC unreliable with few samples
            return
        try:
            auc = roc_auc_score(np.asarray(y_test)[m], np.asarray(preds)[m])
        except ValueError:
            return
        results.append({
            "subgroup": name,
            "n": int(n),
            "n_positive": n_pos,
            "roc_auc": round(float(auc), 4),
        })

    # Age groups
    for label, lo, hi in [("<50", 0, 50), ("50-64", 50, 65), ("65-79", 65, 80), ("80+", 80, 200)]:
        mask = (X_test_raw["age_clean"] >= lo) & (X_test_raw["age_clean"] < hi)
        add_subgroup(f"Age {label}", mask)

    # Gender
    for g in ["Male", "Female"]:
        mask = X_test_raw["gender"].astype(str).str.strip() == g
        add_subgroup(f"Gender: {g}", mask)

    # Ethnicity (groups with >= 20 samples)
    eth_str = X_test_raw["ethnicity"].astype(str).str.strip()
    for eth in eth_str.unique():
        if eth in ("nan", "unknown", ""):
            continue
        mask = eth_str == eth
        if mask.sum() >= 20:
            add_subgroup(f"Ethnicity: {eth}", mask)

    return results

# X_test has same index as original; get raw demographics
X_test_raw = X.loc[X_test.index].copy()
fairness_results = compute_fairness_metrics(X_test_raw, y_test.values, rf_preds)
logistic_fairness_results = compute_fairness_metrics(X_test_raw, y_test.values, log_preds)

# -----------------------------
# Calibration: predicted vs observed risk
# -----------------------------
frac_pos, mean_pred = calibration_curve(
    np.asarray(y_test), rf_preds, n_bins=10, strategy="uniform"
)
calibration_data = {
    "mean_predicted": [round(float(x), 4) for x in mean_pred],
    "fraction_positive": [round(float(x), 4) for x in frac_pos],
}
frac_pos_lr, mean_pred_lr = calibration_curve(
    np.asarray(y_test), log_preds, n_bins=10, strategy="uniform"
)
logistic_calibration = {
    "mean_predicted": [round(float(x), 4) for x in mean_pred_lr],
    "fraction_positive": [round(float(x), 4) for x in frac_pos_lr],
}

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
    "n_estimators": int(rf_model.named_steps["model"].n_estimators),
    "best_params": {k.replace("model__", ""): str(v) for k, v in search.best_params_.items()},
    "fairness": fairness_results,
    "logistic_fairness": logistic_fairness_results,
    "calibration": calibration_data,
    "logistic_calibration": logistic_calibration,
}
with open(model_dir / "model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved to {model_dir / 'model_metrics.json'}")
print("\nFairness (stratified ROC-AUC on test set):")
for r in fairness_results:
    print(f"  {r['subgroup']}: n={r['n']}, n_pos={r['n_positive']}, ROC-AUC={r['roc_auc']:.3f}")
print("\nCalibration: mean predicted vs fraction positive (10 bins) saved to model_metrics.json")