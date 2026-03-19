# ICU Mortality Risk Predictor

**[Live demo](https://clinical-mortality-ml-ja.streamlit.app/)**

A machine learning application that predicts ICU mortality risk from patient vitals and lab values, with SHAP-based explainability for interpretable predictions.

## Overview

This project uses the [eICU Collaborative Research Database](https://eicu-crd.mit.edu/) to train a Random Forest classifier that estimates mortality risk for ICU patients. The Streamlit app allows users to:

- Enter patient demographics and clinical values (vitals, labs)
- Get a predicted mortality risk percentage
- Explore per-patient feature contributions (SHAP)
- View global feature importance across example patients

**Disclaimer:** This project is for educational and portfolio demonstration purposes only. It is not intended for clinical use.

## Key Results

| Metric | Value |
|--------|-------|
| Test ROC-AUC | 0.914 |
| 5-fold CV ROC-AUC | 0.915 ± 0.019 |
| Training samples | 3,486 |
| Test samples | 872 |
| Class balance (train) | 294 positive / 3,192 negative (~8.4% mortality) |
| Features (post-preprocessing) | 23 |

**Model:** Random Forest (hyperparameters tuned via RandomizedSearchCV, 5-fold CV ROC-AUC). Preprocessing: `ColumnTransformer` with `OneHotEncoder` for categoricals (gender, ethnicity) and `SimpleImputer(strategy="median")` for numerics.

**Top drivers of risk** (by mean |SHAP|): sodium (min/max), heart rate, creatinine, systolic BP, age, WBC.

**APACHE comparison:** The app includes a scatter plot comparing model predictions with eICU's built-in APACHE predicted ICU mortality; correlation is computed on a sample from the training data.

**Fairness:** Stratified ROC-AUC by age group, gender, and ethnicity (see Model Info tab and [docs/MODEL_CARD.md](docs/MODEL_CARD.md)).

**Calibration:** Calibration curve (predicted vs observed risk) on the test set; see Model Info tab.

**Input validation:** Vitals and labs are validated against clinically plausible ranges before prediction; min/max pairs (e.g., sodium, creatinine, WBC) must satisfy min ≤ max.

## Screenshots

| [Mortality Prediction](docs/screenshots/prediction.png) | [Patient Explainability](docs/screenshots/explainability.png) | [Model Info](docs/screenshots/model_info.png) |
|--------------------------------------------------------|--------------------------------------------------------------|-----------------------------------------------|
| Enter vitals and get risk %                            | See which factors drive each prediction                      | ROC-AUC, training stats, disclaimer           |

| [Global Feature Importance](docs/screenshots/global_importance.png) | [What-If Analysis](docs/screenshots/whatif.png) |
|---------------------------------------------------------------------|---------------------------------------------------|
| Mean contribution across example patients                            | Risk vs. feature sensitivity                      |

## Tech Stack

- **Python 3.9**
- **scikit-learn** – Random Forest, preprocessing pipeline
- **SHAP** – model explainability
- **Streamlit** – web interface
- **pandas** – data handling

## Setup

### 1. Create conda environment

```bash
conda env create -f environment.conda.yml
conda activate icu_mortality
```

### 2. Data

**Option A – Use pre-processed data (recommended):**  
The repo includes `data/processed/mortality_features.csv`. Skip to step 3.

**Option B – Build from eICU database:**  
Place the eICU SQLite database at `data/raw/eicu_v2_0_1.sqlite3` and run the feature build script (see [Data Pipeline](#data-pipeline)).

### 3. Train the model

```bash
python src/models/train_model.py
```

This creates `models/mortality_model.pkl` and `models/model_metrics.json`.

### 4. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### 5. Deploy to Streamlit Community Cloud (optional)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click "New app", select your repo, set main file to `app.py`.
4. Ensure `models/mortality_model.pkl` and `models/model_metrics.json` are committed (required for deployment).
5. Deploy; add the live URL to your portfolio.

### 6. Run tests (optional)

```bash
python -m pytest tests/test_model.py -v
```

### Alternative: Run with Docker

Ensure `models/` and `data/processed/` exist, then:

```bash
docker-compose up --build
```

Open http://localhost:8501.

**Alternative: pip install**

```bash
pip install -r requirements.txt
```

## Data Pipeline

```
eICU SQLite DB → SQL queries → mortality_features.csv → train_model.py → mortality_model.pkl
```

See `scripts/build_features.py` (or `notebooks/explore_data.ipynb`) for the ETL steps that produce the processed CSV from the raw eICU database.

## Project Structure

```
clinical-mortality-ml/
├── app.py                 # Streamlit application
├── requirements.txt       # pip dependencies
├── Dockerfile             # Docker image
├── docker-compose.yml     # One-command run
├── .streamlit/            # Streamlit config
├── scripts/
│   └── build_features.py   # ETL: eICU SQLite → mortality_features.csv
├── src/
│   ├── config.py          # Paths and config
│   ├── database.py        # SQLite helpers
│   └── models/
│       └── train_model.py # Training pipeline
├── data/
│   ├── raw/               # eICU SQLite (if used)
│   └── processed/         # mortality_features.csv
├── models/                 # Trained model (created by train)
├── tests/                  # Unit tests
│   └── test_model.py
├── notebooks/              # Exploration and ETL
└── environment.conda.yml # Conda dependencies (optional; Streamlit Cloud uses requirements.txt)
```

## Model

See [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for intended use, limitations, and ethical considerations.

- **Algorithm:** Random Forest (200 trees, class-balanced)
- **Features:** Age, gender, ethnicity, vitals (heart rate, BP), labs (creatinine, WBC, sodium), APACHE scores
- **Output:** Probability of ICU mortality (0–100%)

## License

MIT. The eICU dataset has its own license; ensure compliance when using it.
