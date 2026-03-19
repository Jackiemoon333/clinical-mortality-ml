"""Input validation for clinical inputs. Used by app and tests."""

INPUT_RANGES = {
    "age": (18, 120),
    "max_heart_rate": (20, 250),
    "min_systolic_bp": (40, 250),
    "avg_diastolic_bp": (20, 150),
    "max_wbc": (0.5, 100),
    "min_wbc": (0.5, 100),
    "avg_wbc": (0.5, 100),
    "max_creatinine": (0.2, 25),
    "min_creatinine": (0.2, 25),
    "avg_creatinine": (0.2, 25),
    "max_sodium": (100, 180),
    "min_sodium": (100, 180),
    "avg_sodium": (100, 180),
}


def validate_inputs(row):
    """Validate inputs are within clinically plausible ranges and logically consistent.
    Returns (is_valid, list of error messages)."""
    errors = []
    for key, (lo, hi) in INPUT_RANGES.items():
        val = row.get(key)
        if val is not None and not (lo <= float(val) <= hi):
            label = key.replace("_", " ").title()
            errors.append(f"{label}: {val} outside plausible range ({lo}–{hi})")
    if row.get("min_sodium", 0) > row.get("max_sodium", 0):
        errors.append("Min sodium cannot exceed max sodium")
    if row.get("min_creatinine", 0) > row.get("max_creatinine", 0):
        errors.append("Min creatinine cannot exceed max creatinine")
    if row.get("min_wbc", 0) > row.get("max_wbc", 0):
        errors.append("Min WBC cannot exceed max WBC")
    return (len(errors) == 0, errors)
