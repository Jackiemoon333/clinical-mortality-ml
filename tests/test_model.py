"""Unit tests for the mortality prediction model.

Run with: python -m pytest tests/test_model.py -v
Or: python tests/test_model.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation import validate_inputs

EXPECTED_COLUMNS = [
    "age", "gender", "ethnicity", "max_heart_rate", "min_systolic_bp", "avg_diastolic_bp",
    "max_wbc", "min_wbc", "avg_wbc", "max_creatinine", "min_creatinine", "avg_creatinine",
    "max_sodium", "min_sodium", "avg_sodium", "num_unique_meds", "vasopressin",
    "norepinephrine", "dopamine", "acutephysiologyscore", "apachescore",
    "predictedicumortality", "predictedhospitalmortality"
]


def create_sample_input():
    """Create a minimal valid input DataFrame for the model."""
    return pd.DataFrame([{
        "age": 65.0,
        "gender": "Male",
        "ethnicity": "Caucasian",
        "max_heart_rate": 90.0,
        "min_systolic_bp": 110.0,
        "avg_diastolic_bp": 70.0,
        "max_wbc": 8.0,
        "min_wbc": 5.5,
        "avg_wbc": 7.0,
        "max_creatinine": 1.0,
        "min_creatinine": 0.8,
        "avg_creatinine": 0.9,
        "max_sodium": 140.0,
        "min_sodium": 135.0,
        "avg_sodium": 137.5,
        "num_unique_meds": 0,
        "vasopressin": 0,
        "norepinephrine": 0,
        "dopamine": 0,
        "acutephysiologyscore": 0,
        "apachescore": 0,
        "predictedicumortality": 0.0,
        "predictedhospitalmortality": 0.0,
    }])


def test_load_model():
    """Model file exists and can be loaded."""
    model_path = PROJECT_ROOT / "models" / "mortality_model.pkl"
    assert model_path.exists(), f"Model not found at {model_path}"
    model = joblib.load(model_path)
    assert model is not None
    assert hasattr(model, "predict_proba")


def test_prediction_shape():
    """Prediction returns correct shape."""
    model_path = PROJECT_ROOT / "models" / "mortality_model.pkl"
    if not model_path.exists():
        return  # Skip if model not trained
    model = joblib.load(model_path)
    X = create_sample_input()
    proba = model.predict_proba(X)
    assert proba.ndim == 2
    assert proba.shape[0] == 1
    assert proba.shape[1] == 2  # binary classification


def test_prediction_values():
    """Prediction probabilities are valid (0–1)."""
    model_path = PROJECT_ROOT / "models" / "mortality_model.pkl"
    if not model_path.exists():
        return
    model = joblib.load(model_path)
    X = create_sample_input()
    proba = model.predict_proba(X)[0]
    assert np.all(proba >= 0) and np.all(proba <= 1)
    assert np.isclose(proba.sum(), 1.0)


def test_validate_inputs_accepts_valid():
    """Valid inputs pass validation."""
    row = create_sample_input().iloc[0].to_dict()
    valid, errors = validate_inputs(row)
    assert valid, f"Expected valid, got errors: {errors}"
    assert len(errors) == 0


def test_validate_inputs_rejects_out_of_range():
    """Out-of-range values fail validation."""
    row = create_sample_input().iloc[0].to_dict()
    row["age"] = 200  # outside 18-120
    valid, errors = validate_inputs(row)
    assert not valid
    assert any("age" in e.lower() or "Age" in e for e in errors)


def test_validate_inputs_rejects_min_exceeds_max():
    """Min > max for paired values fails validation."""
    row = create_sample_input().iloc[0].to_dict()
    row["min_sodium"] = 150
    row["max_sodium"] = 130
    valid, errors = validate_inputs(row)
    assert not valid
    assert any("sodium" in e.lower() for e in errors)


if __name__ == "__main__":
    import unittest
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
