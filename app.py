import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import shap
import matplotlib.pyplot as plt

st.title("ICU Mortality Risk Predictor")
st.markdown("Enter patient vitals and lab values to estimate ICU mortality risk.")

# Clinical disclaimer
st.caption(
    "For educational and demonstration purposes only. Not for clinical use. "
    "Data from the [eICU Collaborative Research Database](https://eicu-crd.mit.edu/) "
    "([citation](https://www.nature.com/articles/sdata2018178))."
)

# ---- Load models ----
BASE_DIR = Path(__file__).resolve().parent
model_rf = joblib.load(BASE_DIR / "models" / "mortality_model.pkl")
model_lr = joblib.load(BASE_DIR / "models" / "logistic_model.pkl")

# Expected columns for the model (must match training data)
EXPECTED_COLUMNS = [
    "age", "gender", "ethnicity", "max_heart_rate", "min_systolic_bp", "avg_diastolic_bp",
    "max_wbc", "min_wbc", "avg_wbc", "max_creatinine", "min_creatinine", "avg_creatinine",
    "max_sodium", "min_sodium", "avg_sodium", "num_unique_meds", "vasopressin",
    "norepinephrine", "dopamine", "acutephysiologyscore", "apachescore",
    "predictedicumortality", "predictedhospitalmortality"
]

# Human-readable display names for SHAP features
FEATURE_DISPLAY_NAMES = {
    "gender_Male": "Gender: Male",
    "gender_Female": "Gender: Female",
    "gender_nan": "Gender (unknown)",
    "ethnicity_Caucasian": "Ethnicity: Caucasian",
    "ethnicity_African American": "Ethnicity: African American",
    "ethnicity_Hispanic": "Ethnicity: Hispanic",
    "ethnicity_Asian": "Ethnicity: Asian",
    "ethnicity_Other": "Ethnicity: Other",
    "ethnicity_Other/Unknown": "Ethnicity: Other/Unknown",
    "ethnicity_Native American": "Ethnicity: Native American",
    "ethnicity_nan": "Ethnicity (unknown)",
    "age": "Age (years)",
    "max_heart_rate": "Max Heart Rate (bpm)",
    "min_systolic_bp": "Min Systolic BP (mmHg)",
    "avg_diastolic_bp": "Avg Diastolic BP (mmHg)",
    "max_wbc": "Max WBC (K/uL)",
    "min_wbc": "Min WBC (K/uL)",
    "avg_wbc": "Avg WBC (K/uL)",
    "max_creatinine": "Max Creatinine (mg/dL)",
    "min_creatinine": "Min Creatinine (mg/dL)",
    "avg_creatinine": "Avg Creatinine (mg/dL)",
    "max_sodium": "Max Sodium (mEq/L)",
    "min_sodium": "Min Sodium (mEq/L)",
    "avg_sodium": "Avg Sodium (mEq/L)",
    "num_unique_meds": "Number of Unique Meds",
    "vasopressin": "Vasopressin",
    "norepinephrine": "Norepinephrine",
    "dopamine": "Dopamine",
    "acutephysiologyscore": "Acute Physiology Score",
    "apachescore": "APACHE Score",
    "predictedicumortality": "Predicted ICU Mortality",
    "predictedhospitalmortality": "Predicted Hospital Mortality",
}

from src.validation import validate_inputs


def format_feature_name(name):
    """Map internal SHAP feature name to human-readable display label."""
    return FEATURE_DISPLAY_NAMES.get(name, name.replace("_", " ").title())

def get_patient_values_for_display(feature_names, patient_df):
    """Map internal feature names to the patient's actual values for display."""
    row = patient_df.iloc[0]
    values = []
    for name in feature_names:
        if name.startswith("gender_"):
            cat = name.replace("gender_", "")
            values.append("Yes" if str(row["gender"]) == cat else "No")
        elif name.startswith("ethnicity_"):
            cat = name.replace("ethnicity_", "")
            values.append("Yes" if str(row["ethnicity"]) == cat else "No")
        elif name in patient_df.columns:
            val = row[name]
            if isinstance(val, float) and val == int(val):
                values.append(str(int(val)))
            else:
                values.append(f"{val:.2f}" if isinstance(val, float) else str(val))
        else:
            values.append("—")
    return values


def build_lr_coefficient_rows(log_model, patient_df, feature_names):
    """Build coefficient table rows for Logistic Regression. Aggregates one-hots like build_shap_table_rows."""
    X_t = log_model.named_steps["preprocess"].transform(patient_df)
    if X_t.ndim == 1:
        X_t = X_t.reshape(1, -1)
    x_row = X_t[0]
    lr = log_model.named_steps["model"]
    coef = lr.coef_.ravel()
    n_features = min(len(coef), len(x_row), len(feature_names))
    feature_names = feature_names[:n_features]
    # Sanity check: coef @ x_row should approximate decision_function (minus intercept)
    dot_prod = float(np.dot(coef[:n_features], x_row[:n_features]))
    expected = float(lr.decision_function(X_t)[0]) - float(lr.intercept_[0])
    assert abs(dot_prod - expected) < 1e-5, f"Feature alignment mismatch: dot={dot_prod:.4f} vs expected={expected:.4f}"
    row = patient_df.iloc[0]
    seen = {}
    for i, fn in enumerate(feature_names):
        contrib = float(coef[i]) * float(x_row[i])
        if fn.startswith("gender_"):
            key, display = "Gender", "Gender"
            if str(row["gender"]).strip() == fn.replace("gender_", "").strip():
                seen[key] = {"factor": display, "value": str(row["gender"]), "contribution": contrib}
        elif fn.startswith("ethnicity_"):
            key, display = "Ethnicity", "Ethnicity"
            patient_eth = str(row["ethnicity"]).strip()
            if patient_eth == fn.replace("ethnicity_", "").strip() or (patient_eth == "Other" and "Other" in fn):
                seen[key] = {"factor": display, "value": str(row["ethnicity"]), "contribution": contrib}
        else:
            display = format_feature_name(fn)
            val = row.get(fn, None)
            val_str = f"{val:.2f}" if isinstance(val, (float, np.floating)) and val == val else str(val) if val is not None else "—"
            seen[fn] = {"factor": display, "value": val_str, "contribution": contrib}
    return list(seen.values())


def build_shap_table_rows(feature_names, shap_1d, patient_df):
    """Build table rows: aggregate gender/ethnicity one-hots into single rows, one row per numeric feature."""
    row = patient_df.iloc[0]
    seen = {}
    for fn, sv in zip(feature_names, shap_1d):
        sv_val = float(np.asarray(sv).ravel()[0])
        if fn.startswith("gender_"):
            key, display = "Gender", "Gender"
            val_str = str(row["gender"])
            cat = fn.replace("gender_", "")
            if cat in ("nan", "unknown", ""):
                continue
            if str(row["gender"]).strip() == cat.strip():
                seen[key] = {"factor": display, "value": val_str, "contribution": sv_val}
        elif fn.startswith("ethnicity_"):
            key, display = "Ethnicity", "Ethnicity"
            val_str = str(row["ethnicity"])
            cat = fn.replace("ethnicity_", "")
            if cat in ("nan", "unknown", ""):
                continue
            patient_eth = str(row["ethnicity"]).strip()
            if patient_eth == cat.strip() or (patient_eth == "Other" and "Other" in cat):
                seen[key] = {"factor": display, "value": val_str, "contribution": sv_val}
        else:
            display = format_feature_name(fn)
            val = row.get(fn, None)
            try:
                is_na = val is None or (isinstance(val, (float, np.floating)) and (val != val))
            except (TypeError, ValueError):
                is_na = val is None
            if is_na:
                val_str = "—"
            elif isinstance(val, (int, np.integer)):
                val_str = str(int(val))
            elif isinstance(val, (float, np.floating)):
                val_str = str(int(val)) if val == int(val) else f"{val:.2f}"
            else:
                val_str = str(val)
            seen[fn] = {"factor": display, "value": val_str, "contribution": sv_val}
    return list(seen.values())

def get_transformed_feature_names(model=None, n_features=None):
    """Get feature names from the fitted preprocessor (canonical order).
    Uses actual column order from transformers_ to match model output."""
    model = model or model_rf
    preprocess = model.named_steps["preprocess"]
    cat_cols = ["gender", "ethnicity"]
    ohe = preprocess.named_transformers_["cat"]
    cat_feature_names = ohe.get_feature_names_out(cat_cols)
    # Num: use ACTUAL order from fitted transformer (not EXPECTED_COLUMNS)
    num_cols = [c for c in EXPECTED_COLUMNS if c not in cat_cols]
    for name, trans, cols in preprocess.transformers_:
        if name == "num":
            num_cols = list(cols)
            break
    names = list(cat_feature_names) + num_cols
    if n_features is not None:
        names = names[:n_features]
    return names

# ---- Tabs ----
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Mortality Prediction",
    "Patient Explainability",
    "Global Feature Importance",
    "Model Info",
    "What-If Analysis"
])

# ---- Example patients ----
example_patients = {
    "Patient 1": {"age": 45.0, "max_sodium": 140.0, "min_sodium": 135.0, "max_heart_rate": 80.0,
                  "min_systolic_bp": 110.0, "avg_diastolic_bp": 70.0, "max_creatinine": 0.9,
                  "min_creatinine": 0.7, "avg_creatinine": 0.8, "max_wbc": 7.0,
                  "min_wbc": 5.5, "avg_wbc": 6.0, "gender": "Female", "ethnicity": "Caucasian"},
    "Patient 2": {"age": 50.0, "max_sodium": 138.0, "min_sodium": 134.0, "max_heart_rate": 85.0,
                  "min_systolic_bp": 115.0, "avg_diastolic_bp": 72.0, "max_creatinine": 1.0,
                  "min_creatinine": 0.8, "avg_creatinine": 0.9, "max_wbc": 8.0,
                  "min_wbc": 6.0, "avg_wbc": 7.0, "gender": "Male", "ethnicity": "Hispanic"},
    "Patient 3": {"age": 65.0, "max_sodium": 145.0, "min_sodium": 130.0, "max_heart_rate": 100.0,
                  "min_systolic_bp": 100.0, "avg_diastolic_bp": 65.0, "max_creatinine": 1.5,
                  "min_creatinine": 1.0, "avg_creatinine": 1.2, "max_wbc": 12.0,
                  "min_wbc": 7.0, "avg_wbc": 9.0, "gender": "Female", "ethnicity": "Asian"},
    "Patient 4": {"age": 70.0, "max_sodium": 142.0, "min_sodium": 128.0, "max_heart_rate": 95.0,
                  "min_systolic_bp": 105.0, "avg_diastolic_bp": 68.0, "max_creatinine": 1.4,
                  "min_creatinine": 1.0, "avg_creatinine": 1.2, "max_wbc": 11.0,
                  "min_wbc": 6.5, "avg_wbc": 9.0, "gender": "Male", "ethnicity": "African American"},
    "Patient 5": {"age": 68.0, "max_sodium": 144.0, "min_sodium": 129.0, "max_heart_rate": 105.0,
                  "min_systolic_bp": 102.0, "avg_diastolic_bp": 66.0, "max_creatinine": 1.6,
                  "min_creatinine": 1.1, "avg_creatinine": 1.3, "max_wbc": 13.0,
                  "min_wbc": 7.0, "avg_wbc": 10.0, "gender": "Female", "ethnicity": "Other"},
    "Patient 6": {"age": 85.0, "max_sodium": 150.0, "min_sodium": 125.0, "max_heart_rate": 120.0,
                  "min_systolic_bp": 85.0, "avg_diastolic_bp": 55.0, "max_creatinine": 3.0,
                  "min_creatinine": 2.0, "avg_creatinine": 2.5, "max_wbc": 20.0,
                  "min_wbc": 10.0, "avg_wbc": 15.0, "gender": "Male", "ethnicity": "Caucasian"},
    "Patient 7": {"age": 90.0, "max_sodium": 148.0, "min_sodium": 126.0, "max_heart_rate": 115.0,
                  "min_systolic_bp": 88.0, "avg_diastolic_bp": 58.0, "max_creatinine": 2.8,
                  "min_creatinine": 2.0, "avg_creatinine": 2.4, "max_wbc": 18.0,
                  "min_wbc": 9.0, "avg_wbc": 14.0, "gender": "Female", "ethnicity": "Hispanic"},
    "Patient 8": {"age": 88.0, "max_sodium": 149.0, "min_sodium": 127.0, "max_heart_rate": 118.0,
                  "min_systolic_bp": 87.0, "avg_diastolic_bp": 57.0, "max_creatinine": 2.9,
                  "min_creatinine": 2.1, "avg_creatinine": 2.5, "max_wbc": 19.0,
                  "min_wbc": 9.5, "avg_wbc": 14.5, "gender": "Male", "ethnicity": "Asian"}
}

# ---- Select example patient ----
selected_patient = st.selectbox("Select Example Patient", ["Custom Input"] + list(example_patients.keys()))
patient_data = example_patients[selected_patient] if selected_patient != "Custom Input" else {}

# ---- Model selector ----
model_choice = st.radio("Model", ["Random Forest", "Logistic Regression"], horizontal=True, index=0)
model = model_rf if model_choice == "Random Forest" else model_lr

# ---- Input widgets ----
age = st.number_input("Age", 18.0, 100.0, patient_data.get("age", 65.0))
max_sodium = st.number_input("Max Sodium", 120.0, 180.0, patient_data.get("max_sodium", 140.0))
min_sodium = st.number_input("Min Sodium", 120.0, 180.0, patient_data.get("min_sodium", 135.0))
max_heart_rate = st.number_input("Max Heart Rate", 40.0, 200.0, patient_data.get("max_heart_rate", 90.0))
min_systolic_bp = st.number_input("Min Systolic BP", 60.0, 200.0, patient_data.get("min_systolic_bp", 110.0))
avg_diastolic_bp = st.number_input("Avg Diastolic BP", 40.0, 120.0, patient_data.get("avg_diastolic_bp", 70.0))
max_creatinine = st.number_input("Max Creatinine", 0.3, 10.0, patient_data.get("max_creatinine", 1.0))
min_creatinine = st.number_input("Min Creatinine", 0.3, 10.0, patient_data.get("min_creatinine", 0.8))
avg_creatinine = st.number_input("Avg Creatinine", 0.3, 10.0, patient_data.get("avg_creatinine", 0.9))
max_wbc = st.number_input("Max WBC", 1.0, 50.0, patient_data.get("max_wbc", 8.0))
min_wbc = st.number_input("Min WBC", 1.0, 50.0, patient_data.get("min_wbc", 5.5))
avg_wbc = st.number_input("Avg WBC", 1.0, 50.0, patient_data.get("avg_wbc", 7.0))
gender = st.selectbox("Gender", ["Male", "Female"], index=["Male","Female"].index(patient_data.get("gender","Male")))
ethnicity = st.selectbox("Ethnicity", ["Caucasian","African American","Hispanic","Asian","Other"], 
                         index=["Caucasian","African American","Hispanic","Asian","Other"].index(patient_data.get("ethnicity","Caucasian")))

# ---- Build DataFrame for prediction ----
input_data = pd.DataFrame([{
    "age": age,
    "max_sodium": max_sodium,
    "min_sodium": min_sodium,
    "avg_sodium": (max_sodium + min_sodium) / 2,
    "max_heart_rate": max_heart_rate,
    "min_systolic_bp": min_systolic_bp,
    "avg_diastolic_bp": avg_diastolic_bp,
    "max_creatinine": max_creatinine,
    "min_creatinine": min_creatinine,
    "avg_creatinine": avg_creatinine,
    "max_wbc": max_wbc,
    "min_wbc": min_wbc,
    "avg_wbc": avg_wbc,
    "gender": gender,
    "ethnicity": ethnicity,
    "num_unique_meds": 0,
    "vasopressin": 0,
    "norepinephrine": 0,
    "dopamine": 0,
    "acutephysiologyscore": 0,
    "apachescore": 0,
    "predictedicumortality": 0.0,
    "predictedhospitalmortality": 0.0
}])

# ---- Tab 1: Mortality Prediction ----
with tab1:
    if st.button("Predict Mortality Risk"):
        row = input_data.iloc[0].to_dict()
        valid, errors = validate_inputs(row)
        if not valid:
            for err in errors:
                st.error(err)
            st.info("Adjust values to clinically plausible ranges and try again.")
        else:
            prob = model.predict_proba(input_data)[0][1]
            risk_percent = prob * 100
            if model_choice == "Random Forest":
                trees = model.named_steps["model"].estimators_
                tree_probs = np.array([t.predict_proba(model.named_steps["preprocess"].transform(input_data))[:, 1] for t in trees])
                tree_probs = tree_probs.ravel()
                std_p = np.clip(np.std(tree_probs) * 1.96, 0, 1)
                ci_lo = np.clip(prob - std_p, 0, 1) * 100
                ci_hi = np.clip(prob + std_p, 0, 1) * 100
                ci_str = f" (95% CI: {ci_lo:.1f}%–{ci_hi:.1f}%)"
            else:
                ci_str = ""
            st.progress(int(risk_percent))
            if risk_percent < 5:
                st.success(f"Low Risk: {risk_percent:.2f}%{ci_str}")
            elif risk_percent < 20:
                st.warning(f"Moderate Risk: {risk_percent:.2f}%{ci_str}")
            else:
                st.error(f"High Risk: {risk_percent:.2f}%{ci_str}")
            st.session_state["last_patient_data"] = input_data
            st.session_state["last_risk_percent"] = risk_percent
            st.session_state["last_model_choice"] = model_choice

# ---- Tab 2: Patient Explainability ----
with tab2:
    st.subheader("Patient Feature Contributions")
    if "last_patient_data" in st.session_state:
        patient_data = st.session_state["last_patient_data"]
        risk_pct = st.session_state.get("last_risk_percent", 0)
        last_model = st.session_state.get("last_model_choice", "Random Forest")

        feature_names = get_transformed_feature_names(model_lr if last_model == "Logistic Regression" else model_rf)
        n_features = len(feature_names)

        st.markdown(f"**Predicted mortality risk: {risk_pct:.1f}%** ({last_model})")
        st.markdown("How each factor contributes to this prediction:")

        if last_model == "Random Forest":
            X_transformed = model_rf.named_steps['preprocess'].transform(patient_data)
            if X_transformed.ndim == 1:
                X_transformed = X_transformed.reshape(1, -1)
            inner_model = model_rf.named_steps['model']
            explainer = shap.TreeExplainer(inner_model)
            shap_values = explainer.shap_values(X_transformed)
            shap_for_class1 = shap_values[1] if isinstance(shap_values, list) else shap_values
            shap_raw = shap_for_class1[0] if shap_for_class1.ndim >= 2 else shap_for_class1
            arr = np.asarray(shap_raw)
            if arr.ndim == 3:
                shap_1d = np.sum(arr, axis=-1).ravel()
            elif arr.ndim == 2:
                shap_1d = arr[0] if arr.shape[0] == 1 else arr[:, 0] if arr.shape[1] == 2 else arr[0]
            else:
                shap_1d = arr
            shap_1d = np.asarray(shap_1d).ravel()
            table_rows = build_shap_table_rows(feature_names, shap_1d, patient_data)

            table_data = [
                {"Factor": r["factor"], "Your value": r["value"], "Contribution": f"{r['contribution']:.4f}",
                 "Effect": "Increases risk" if r["contribution"] > 0 else "Decreases risk", "_sort_key": abs(r["contribution"])}
                for r in table_rows
            ]
            table_data.sort(key=lambda r: r["_sort_key"], reverse=True)
            table_df = pd.DataFrame([{k: v for k, v in r.items() if k != "_sort_key"} for r in table_data[:15]])
            st.dataframe(table_df, use_container_width=True)

            st.markdown("**Feature contributions (bar chart)**")
            display_names = [format_feature_name(fn) for fn in feature_names]
            shap_2d = shap_1d.reshape(1, -1)
            plt.figure(figsize=(10, 5))
            shap.summary_plot(shap_2d, X_transformed, feature_names=display_names, plot_type="bar", max_display=12, show=False)
            fig = plt.gcf()
            fig.patch.set_facecolor("#0e1117")
            for ax in fig.axes:
                ax.set_facecolor("#0e1117")
                ax.tick_params(colors="white")
                if ax.get_xlabel():
                    ax.xaxis.label.set_color("white")
                if ax.get_ylabel():
                    ax.yaxis.label.set_color("white")
                for spine in ax.spines.values():
                    spine.set_color("white")
                for text in ax.get_yticklabels() + ax.get_xticklabels():
                    text.set_color("white")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
        else:
            table_rows = build_lr_coefficient_rows(model_lr, patient_data, feature_names)
            table_data = [
                {"Factor": r["factor"], "Your value": r["value"], "Contribution (log-odds)": f"{r['contribution']:.4f}",
                 "Effect": "Increases risk" if r["contribution"] > 0 else "Decreases risk", "_sort_key": abs(r["contribution"])}
                for r in table_rows
            ]
            table_data.sort(key=lambda r: r["_sort_key"], reverse=True)
            table_df = pd.DataFrame([{k: v for k, v in r.items() if k != "_sort_key"} for r in table_data[:15]])
            st.dataframe(table_df, use_container_width=True)
            st.caption("Logistic regression coefficients: contribution = coefficient × your value. Positive = increases risk.")

        st.markdown("**Download prediction report**")
        sorted_rows = sorted(table_rows, key=lambda r: -abs(r["contribution"]))[:15]
        top_factors_html = "".join(
            f"<tr><td>{r['factor']}</td><td>{r['value']}</td><td>{r['contribution']:.4f}</td>"
            f"<td>{'Increases risk' if r['contribution'] > 0 else 'Decreases risk'}</td></tr>"
            for r in sorted_rows
        )
        patient_row = patient_data.iloc[0]
        inputs_html = "".join(
            f"<tr><td>{format_feature_name(k) if k in FEATURE_DISPLAY_NAMES else k}</td><td>{patient_row.get(k, '—')}</td></tr>"
            for k in EXPECTED_COLUMNS if k in patient_row
        )
        report_html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>ICU Mortality Prediction Report</title>
<style>body{{font-family:sans-serif;max-width:800px;margin:2em auto;padding:1em}} table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #ccc;padding:8px;text-align:left}} th{{background:#f0f0f0}}</style>
</head>
<body>
<h1>ICU Mortality Prediction Report</h1>
<p><em>For educational purposes only. Not for clinical use.</em></p>
<h2>Prediction</h2>
<p><strong>Predicted mortality risk: {risk_pct:.1f}%</strong></p>
<h2>Patient Inputs</h2>
<table><tr><th>Factor</th><th>Value</th></tr>{inputs_html}</table>
<h2>Top Contributing Factors</h2>
<table><tr><th>Factor</th><th>Your value</th><th>Contribution</th><th>Effect</th></tr>{top_factors_html}</table>
<p><small>Generated by ICU Mortality Risk Predictor. Data from eICU Collaborative Research Database.</small></p>
</body>
</html>"""
        st.download_button("Download prediction report (HTML)", report_html, file_name="prediction_report.html", mime="text/html")
    else:
        st.info("Run a prediction first to see feature contributions.")

# ---- Tab 3: Global Feature Importance ----
with tab3:
    st.subheader("Global Feature Importance Across Example Patients")
    if st.button("Compute Global Feature Importance"):
        feature_names = get_transformed_feature_names(model_lr if model_choice == "Logistic Regression" else model_rf)
        if model_choice == "Random Forest":
            st.caption("Mean |SHAP| across example patients.")
            all_examples = pd.DataFrame(example_patients).T
            all_examples["avg_sodium"] = (all_examples["max_sodium"] + all_examples["min_sodium"]) / 2
            for col in ["num_unique_meds", "vasopressin", "norepinephrine", "dopamine",
                        "acutephysiologyscore", "apachescore", "predictedicumortality", "predictedhospitalmortality"]:
                if col not in all_examples.columns:
                    all_examples[col] = 0.0
            all_examples = all_examples[EXPECTED_COLUMNS]
            X_all_transformed = model_rf.named_steps['preprocess'].transform(all_examples)
            if X_all_transformed.ndim == 1:
                X_all_transformed = X_all_transformed.reshape(1, -1)
            n_features = min(X_all_transformed.shape[1], len(feature_names))
            feature_names = feature_names[:n_features]
            inner_model = model_rf.named_steps['model']
            explainer = shap.TreeExplainer(inner_model)
            shap_values_all = explainer.shap_values(X_all_transformed)
            shap_for_class1 = shap_values_all[1] if isinstance(shap_values_all, list) else shap_values_all
            shap_arr = np.asarray(shap_for_class1)
            if shap_arr.ndim == 3:
                mean_abs_shap = np.mean(np.abs(shap_arr).sum(axis=-1), axis=0)
            else:
                mean_abs_shap = np.mean(np.abs(shap_arr), axis=0)
            mean_abs_shap = mean_abs_shap[:n_features]
            importance_vals = mean_abs_shap
            col_label = "Mean |Contribution|"
        else:
            st.caption("|Coefficient|: for LR, coefficient magnitude = effect per unit change (global by definition).")
            coef = model_lr.named_steps["model"].coef_.ravel()
            n_features = min(len(coef), len(feature_names))
            feature_names = feature_names[:n_features]
            coef = coef[:n_features]
            importance_vals = np.abs(coef)
            col_label = "|Coefficient|"
        order = np.argsort(importance_vals)[::-1]
        table_rows = []
        for rank, idx in enumerate(order[:12], 1):
            table_rows.append({
                "Rank": rank,
                "Factor": format_feature_name(feature_names[idx]),
                col_label: f"{importance_vals[idx]:.4f}"
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
        top_n = 10
        top_indices = order[:top_n]
        chart_df = pd.DataFrame({
            "Factor": [format_feature_name(feature_names[i]) for i in top_indices],
            "Importance": [importance_vals[i] for i in top_indices]
        })
        chart_df = chart_df.iloc[::-1]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(chart_df["Factor"], chart_df["Importance"], color="#1f77b4")
        ax.set_xlabel(col_label)
        ax.set_title("Top Factors by Importance")
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("white")
        for text in ax.get_yticklabels() + ax.get_xticklabels():
            text.set_color("white")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ---- Tab 4: Model Info ----
with tab4:
    st.subheader("Model Performance")
    metrics_path = BASE_DIR / "models" / "model_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        # Show metrics for selected model
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            auc_val = metrics.get("logistic_roc_auc" if model_choice == "Logistic Regression" else "roc_auc", 0)
            st.metric("Test ROC-AUC", f"{auc_val:.3f}")
        with c2:
            if model_choice == "Random Forest":
                cv_mean = metrics.get("cv_roc_auc_mean")
                cv_std = metrics.get("cv_roc_auc_std")
                st.metric("5-fold CV ROC-AUC", f"{cv_mean:.3f} ± {cv_std:.3f}" if cv_mean is not None and cv_std is not None else "—")
            else:
                st.metric("5-fold CV ROC-AUC", "—")
        with c3:
            st.metric("Training samples", metrics.get("n_train", "—"))
        with c4:
            st.metric("Test samples", metrics.get("n_test", "—"))
        log_auc = metrics.get("logistic_roc_auc")
        rf_auc = metrics.get("roc_auc")
        if log_auc is not None and rf_auc is not None:
            diff = rf_auc - log_auc
            st.caption(f"Random Forest outperformed Logistic Regression by {diff:.3f} ROC-AUC (RF: {rf_auc:.3f}, LR: {log_auc:.3f}). LR offers direct coefficient interpretability.")
        st.markdown("**Class balance (train):** "
                    f"{metrics.get('class_balance', {}).get('train_pos', '—')} positive / "
                    f"{metrics.get('class_balance', {}).get('train_neg', '—')} negative")
    else:
        st.info("Run `python src/models/train_model.py` to generate model metrics.")
        st.metric("ROC-AUC", "—")
    st.subheader("Model Description")
    if model_choice == "Random Forest":
        best_params = metrics.get("best_params", {}) if metrics_path.exists() else {}
        n_est = metrics.get("n_estimators", 200)
        st.markdown(f"""
        - **Algorithm:** Random Forest ({n_est} trees, class-balanced)
        - **Features:** Age, gender, ethnicity, vitals (heart rate, BP), labs (creatinine, WBC, sodium), APACHE scores
        - **Output:** Probability of ICU mortality (0–100%)
        - **Data:** eICU Collaborative Research Database
        """)
        if best_params:
            with st.expander("Hyperparameters (tuned via RandomizedSearchCV)"):
                st.json(best_params)
    else:
        lr_model = model_lr.named_steps["model"]
        lr_params = {"penalty": getattr(lr_model, "penalty", "l2"), "C": float(getattr(lr_model, "C", 1.0)),
                     "solver": getattr(lr_model, "solver", "lbfgs"), "max_iter": int(getattr(lr_model, "max_iter", 1000))}
        st.markdown("""
        - **Algorithm:** Logistic Regression (class-balanced)
        - **Features:** Age, gender, ethnicity, vitals (heart rate, BP), labs (creatinine, WBC, sodium), APACHE scores
        - **Output:** Probability of ICU mortality (0–100%)
        - **Data:** eICU Collaborative Research Database
        """)
        with st.expander("Hyperparameters"):
            st.json(lr_params)
    # Fairness analysis
    with st.expander("Fairness: stratified performance by subgroup"):
        fairness_key = "logistic_fairness" if model_choice == "Logistic Regression" else "fairness"
        fairness = metrics.get(fairness_key, []) if metrics_path.exists() else []
        if fairness:
            fairness_df = pd.DataFrame(fairness)
            fairness_df = fairness_df.rename(columns={"subgroup": "Subgroup", "n": "n", "n_positive": "Deaths", "roc_auc": "ROC-AUC"})
            st.dataframe(fairness_df[["Subgroup", "n", "Deaths", "ROC-AUC"]], use_container_width=True)
            st.caption(
                "ROC-AUC computed on test set within each subgroup. "
                "Small subgroups (n < 20 or < 3 deaths) are excluded. "
                "Large gaps between subgroups may indicate differential performance."
            )
        else:
            st.caption("Run `python src/models/train_model.py` to generate fairness metrics.")

    # Calibration plot
    with st.expander("Calibration: predicted vs observed risk"):
        cal_key = "logistic_calibration" if model_choice == "Logistic Regression" else "calibration"
        cal = metrics.get(cal_key, {}) if metrics_path.exists() else {}
        if cal and "mean_predicted" in cal and "fraction_positive" in cal:
            mean_pred = cal["mean_predicted"]
            frac_pos = cal["fraction_positive"]
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
            ax.plot(mean_pred, frac_pos, "s-", color="#1f77b4", label="Model")
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Fraction of positives (observed)")
            ax.set_title("Calibration Curve (test set)")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(loc="lower right")
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#0e1117")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            for spine in ax.spines.values():
                spine.set_color("white")
            for text in ax.get_yticklabels() + ax.get_xticklabels():
                text.set_color("white")
            ax.legend(facecolor="#0e1117", edgecolor="white", labelcolor="white")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.caption(
                "Points near the diagonal indicate well-calibrated probabilities. "
                "Points above the line: model underestimates risk; below: overestimates."
            )
        else:
            st.caption("Run `python src/models/train_model.py` to generate calibration data.")

    st.subheader("Disclaimer")
    st.warning(
        "This tool is for educational and portfolio demonstration only. "
        "It is not validated for clinical use. Do not use for medical decisions."
    )

    # APACHE comparison
    with st.expander("Compare with APACHE scores"):
        data_path = BASE_DIR / "data" / "processed" / "mortality_features.csv"
        if data_path.exists():
            df_apache = pd.read_csv(data_path)
            df_apache = df_apache.dropna(subset=["predictedicumortality"])
            df_apache["predictedicumortality"] = pd.to_numeric(df_apache["predictedicumortality"], errors="coerce")
            df_apache = df_apache.dropna(subset=["predictedicumortality"])
            df_apache = df_apache[
                (df_apache["predictedicumortality"] >= 0) & (df_apache["predictedicumortality"] <= 1)
            ]
            # Clean age to match training ("> 89" -> 90)
            df_apache["age"] = df_apache["age"].replace("> 89", 90)
            df_apache["age"] = pd.to_numeric(df_apache["age"], errors="coerce")
            df_apache["age"] = df_apache["age"].fillna(df_apache["age"].median())
            if len(df_apache) > 0:
                X_apache = df_apache.drop(columns=["target_mortality", "patientunitstayid"], errors="ignore")
                X_apache = X_apache[EXPECTED_COLUMNS] if all(c in X_apache.columns for c in EXPECTED_COLUMNS) else X_apache
                if len(X_apache.columns) == len(EXPECTED_COLUMNS):
                    sample = X_apache.sample(min(500, len(X_apache)), random_state=42)
                    model_probs = model.predict_proba(sample)[:, 1]
                    apache_probs = df_apache.loc[sample.index, "predictedicumortality"].values
                    corr = np.corrcoef(model_probs, apache_probs)[0, 1] if len(model_probs) > 1 else 0
                    st.markdown(f"**Correlation** between this model's risk and eICU APACHE predicted ICU mortality: **{corr:.3f}**")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(apache_probs * 100, model_probs * 100, alpha=0.5, s=10, color="#1f77b4")
                    ax.plot([0, 100], [0, 100], "k--", alpha=0.5, label="y=x")
                    ax.set_xlabel("APACHE predicted ICU mortality (%)")
                    ax.set_ylabel("This model predicted mortality (%)")
                    ax.set_title("Model vs. APACHE Predicted ICU Mortality")
                    fig.patch.set_facecolor("#0e1117")
                    ax.set_facecolor("#0e1117")
                    ax.tick_params(colors="white")
                    ax.xaxis.label.set_color("white")
                    ax.yaxis.label.set_color("white")
                    ax.title.set_color("white")
                    for spine in ax.spines.values():
                        spine.set_color("white")
                    for text in ax.get_yticklabels() + ax.get_xticklabels():
                        text.set_color("white")
                    ax.legend(facecolor="#0e1117", edgecolor="white", labelcolor="white")
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()
                else:
                    st.caption("Feature columns do not match; APACHE comparison skipped.")
            else:
                st.caption("No APACHE predictions in data.")
        else:
            st.caption("Data file not found for APACHE comparison.")

# ---- Tab 5: What-If Analysis ----
SENSITIVITY_FEATURES = {
    "age": ("Age (years)", 18, 100),
    "max_creatinine": ("Max Creatinine (mg/dL)", 0.3, 10.0),
    "avg_creatinine": ("Avg Creatinine (mg/dL)", 0.3, 10.0),
    "max_heart_rate": ("Max Heart Rate (bpm)", 40, 200),
    "min_systolic_bp": ("Min Systolic BP (mmHg)", 60, 200),
    "avg_diastolic_bp": ("Avg Diastolic BP (mmHg)", 40, 120),
    "max_wbc": ("Max WBC (K/uL)", 1.0, 50.0),
    "avg_wbc": ("Avg WBC (K/uL)", 1.0, 50.0),
    "max_sodium": ("Max Sodium (mEq/L)", 120, 180),
    "min_sodium": ("Min Sodium (mEq/L)", 120, 180),
}
with tab5:
    st.subheader("What if I change X?")
    st.markdown("Adjust one feature and see how predicted mortality risk changes. Other features stay at current values.")
    base_df = st.session_state.get("last_patient_data", input_data.copy())
    feature_key = st.selectbox(
        "Feature to vary",
        options=list(SENSITIVITY_FEATURES.keys()),
        format_func=lambda k: SENSITIVITY_FEATURES[k][0],
    )
    label, lo, hi = SENSITIVITY_FEATURES[feature_key]
    base_val = float(base_df[feature_key].iloc[0]) if feature_key in base_df.columns else (float(lo) + float(hi)) / 2
    default_lo = max(float(lo), base_val * 0.5)
    default_hi = min(float(hi), base_val * 1.5) if base_val > 0 else float(hi) * 0.5
    if default_lo >= default_hi:
        default_lo, default_hi = float(lo), float(hi)
    val_lo = st.slider(f"Range start ({label})", float(lo), float(hi), default_lo)
    val_hi = st.slider(f"Range end ({label})", float(lo), float(hi), default_hi)
    if val_lo >= val_hi:
        val_lo, val_hi = val_hi, val_lo
    n_points = 30
    values = np.linspace(val_lo, val_hi, n_points)
    risks = []
    for v in values:
        row = base_df.iloc[0].to_dict()
        row[feature_key] = v
        if feature_key in ("max_sodium", "min_sodium"):
            row["avg_sodium"] = (row["max_sodium"] + row["min_sodium"]) / 2
        df_row = pd.DataFrame([row])
        prob = model.predict_proba(df_row)[0][1]
        risks.append(prob * 100)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(values, risks, color="#1f77b4", linewidth=2)
    ax.set_xlabel(label)
    ax.set_ylabel("Predicted mortality risk (%)")
    ax.set_title(f"Risk vs. {label}")
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("white")
    for text in ax.get_yticklabels() + ax.get_xticklabels():
        text.set_color("white")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()