#!/usr/bin/env python3
"""
Build mortality features from eICU SQLite database.

Flow: eICU SQLite -> SQL queries -> processed CSV

Usage:
    python scripts/build_features.py

Requires:
    - data/raw/eicu_v2_0_1.sqlite3 (eICU Collaborative Research Database)
    - Run from project root

Output:
    - data/processed/mortality_features.csv
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Project root (script lives in scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EICU_DB = PROJECT_ROOT / "data" / "raw" / "eicu_v2_0_1.sqlite3"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "mortality_features.csv"


def run_query(query: str, db_path: Path) -> pd.DataFrame:
    """Run SQL query against eICU SQLite and return DataFrame."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def main():
    if not EICU_DB.exists():
        print(f"Error: eICU database not found at {EICU_DB}")
        print("Download from https://eicu-crd.mit.edu/ and place in data/raw/")
        sys.exit(1)

    print("Building features from eICU database...")

    # 1. Patient demographics + vitals + target
    query_patient = """
    SELECT
        p.patientunitstayid,
        p.age,
        p.gender,
        p.ethnicity,
        CASE WHEN p.hospitaldischargestatus = 'Expired' THEN 1 ELSE 0 END AS target_mortality,
        MAX(v.heartrate) AS max_heart_rate,
        MIN(v.systemicsystolic) AS min_systolic_bp,
        AVG(v.systemicdiastolic) AS avg_diastolic_bp
    FROM patient p
    LEFT JOIN vitalPeriodic v ON p.patientunitstayid = v.patientunitstayid
    GROUP BY p.patientunitstayid, p.age, p.gender, p.ethnicity, target_mortality
    """
    df = run_query(query_patient, EICU_DB)

    # 2. Labs (WBC, Creatinine, Sodium)
    query_labs = """
    SELECT
        patientunitstayid,
        MAX(CASE WHEN labname = 'WBC' THEN labresult END) AS max_wbc,
        MIN(CASE WHEN labname = 'WBC' THEN labresult END) AS min_wbc,
        AVG(CASE WHEN labname = 'WBC' THEN labresult END) AS avg_wbc,
        MAX(CASE WHEN labname = 'Creatinine' THEN labresult END) AS max_creatinine,
        MIN(CASE WHEN labname = 'Creatinine' THEN labresult END) AS min_creatinine,
        AVG(CASE WHEN labname = 'Creatinine' THEN labresult END) AS avg_creatinine,
        MAX(CASE WHEN labname = 'Sodium' THEN labresult END) AS max_sodium,
        MIN(CASE WHEN labname = 'Sodium' THEN labresult END) AS min_sodium,
        AVG(CASE WHEN labname = 'Sodium' THEN labresult END) AS avg_sodium
    FROM lab
    GROUP BY patientunitstayid
    """
    df_labs = run_query(query_labs, EICU_DB)
    df = df.merge(df_labs, on="patientunitstayid", how="left")

    # 3. Medications
    query_med = """
    SELECT
        patientunitstayid,
        COUNT(DISTINCT drugname) AS num_unique_meds,
        MAX(CASE WHEN drugname LIKE '%Vasopressin%' THEN 1 ELSE 0 END) AS vasopressin,
        MAX(CASE WHEN drugname LIKE '%Norepinephrine%' THEN 1 ELSE 0 END) AS norepinephrine,
        MAX(CASE WHEN drugname LIKE '%Dopamine%' THEN 1 ELSE 0 END) AS dopamine
    FROM medication
    GROUP BY patientunitstayid
    """
    df_med = run_query(query_med, EICU_DB)
    df = df.merge(df_med, on="patientunitstayid", how="left")

    # 4. APACHE scores
    query_apache = """
    SELECT
        patientunitstayid,
        acutephysiologyscore,
        apachescore,
        predictedicumortality,
        predictedhospitalmortality
    FROM apachepatientresult
    """
    df_apache = run_query(query_apache, EICU_DB)
    df = df.merge(df_apache, on="patientunitstayid", how="left")

    # Deduplicate (apachepatientresult can have multiple rows per stay)
    df = df.drop_duplicates(subset="patientunitstayid")

    # 5. Data cleaning
    df = df.replace("", np.nan)

    # Age
    df["age"] = df["age"].replace("> 89", 90)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["age"] = df["age"].fillna(df["age"].median())

    # Numeric columns
    numeric_cols = [
        "avg_diastolic_bp", "min_systolic_bp", "max_heart_rate",
        "max_wbc", "min_wbc", "avg_wbc",
        "max_creatinine", "min_creatinine", "avg_creatinine",
        "max_sodium", "min_sodium", "avg_sodium",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Vitals: fill missing with median
    vital_cols = ["avg_diastolic_bp", "min_systolic_bp", "max_heart_rate"]
    for col in vital_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Labs: fill missing with median (model uses SimpleImputer at train time too)
    lab_cols = ["max_wbc", "min_wbc", "avg_wbc", "max_creatinine", "min_creatinine",
                "avg_creatinine", "max_sodium", "min_sodium", "avg_sodium"]
    for col in lab_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Medications
    med_cols = ["num_unique_meds", "vasopressin", "norepinephrine", "dopamine"]
    for col in med_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # APACHE scores
    apache_cols = ["acutephysiologyscore", "apachescore",
                   "predictedicumortality", "predictedhospitalmortality"]
    for col in apache_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Categorical
    df["gender"] = df["gender"].fillna(df["gender"].mode().iloc[0])
    df["ethnicity"] = df["ethnicity"].fillna("Unknown")

    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
    print(f"Target balance: {df['target_mortality'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
