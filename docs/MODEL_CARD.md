# Model Card: ICU Mortality Risk Predictor

## Model Details

- **Name:** ICU Mortality Risk Predictor
- **Version:** 1.0
- **Type:** Binary classification (Random Forest)
- **Framework:** scikit-learn
- **Architecture:** Pipeline with ColumnTransformer (OneHotEncoder for categoricals, SimpleImputer for numerics) and RandomForestClassifier. Hyperparameters tuned via RandomizedSearchCV (5-fold CV, ROC-AUC).

## Intended Use

- **Primary use:** Educational and portfolio demonstration of clinical ML and explainability (SHAP).
- **Intended users:** Developers, students, and researchers learning about mortality prediction in critical care.
- **Out-of-scope:** This model is **not** intended for clinical decision-making, triage, or any medical use. It has not been validated for deployment in healthcare settings.

## Limitations

- **Data scope:** Trained on the eICU Collaborative Research Database (2014–2015 US ICU stays). Performance may not generalize to other populations, time periods, or care settings.
- **Feature availability:** Requires vitals, labs, and APACHE scores. Missing or incomplete data are imputed with medians, which may introduce bias.
- **Temporal validity:** Training data is from 2014–2015; clinical practice and outcomes may have changed.
- **Demographic fairness:** Performance may vary across age, gender, and ethnicity subgroups. See Fairness Evaluation below.
- **Explainability:** SHAP values provide local approximations; they do not establish causal relationships.

## Training Data

- **Source:** [eICU Collaborative Research Database](https://eicu-crd.mit.edu/) ([citation](https://www.nature.com/articles/sdata2018178))
- **Target:** Hospital discharge status (Expired = 1, else 0)
- **Features:** Age, gender, ethnicity; vitals (heart rate, systolic/diastolic BP); labs (WBC, creatinine, sodium); medications (vasopressin, norepinephrine, dopamine); APACHE scores
- **Class balance:** Imbalanced (typically ~8–10% positive class). Class weights used during training.
- **Preprocessing:** Age cleaned (>89 → 90); missing values imputed with median; categoricals one-hot encoded.

## Metrics

| Metric   | Value (example) |
|----------|-----------------|
| ROC-AUC  | ~0.85 (see `models/model_metrics.json` after training) |
| n_train  | ~2000 |
| n_test   | ~500  |
| n_features | 22+ (after one-hot encoding) |

*Exact values are written to `models/model_metrics.json` when running `python src/models/train_model.py`.*

## Fairness Evaluation

Stratified ROC-AUC was computed on the held-out test set by age group, gender, and ethnicity. Subgroups with fewer than 20 samples or fewer than 3 positive (death) cases were excluded, as ROC-AUC is unreliable with very small samples.

**Interpretation:** Performance is generally consistent across subgroups (typical ROC-AUC 0.86–0.94). Some variation is expected due to sample size and class imbalance within subgroups. Larger gaps (e.g., >0.05 ROC-AUC) between demographic groups warrant further investigation before deployment. The eICU dataset is US-based and may not represent global diversity; underrepresented ethnicities in the test set may not have sufficient data for reliable subgroup metrics.

See the app's Model Info tab for the full fairness table.

## Calibration

A calibration curve compares mean predicted probability (x-axis) with the observed fraction of positives (y-axis) in each probability bin. A well-calibrated model lies near the diagonal: a predicted 30% risk should correspond to ~30% observed mortality.

**Interpretation:** Random Forest models often exhibit some miscalibration (e.g., overconfident in extreme predictions). The app's Model Info tab includes a calibration plot on the test set. Points above the diagonal indicate underestimation of risk; points below indicate overestimation. For clinical use, post-hoc calibration (e.g., Platt scaling or isotonic regression) could be applied if needed.

## Ethical Considerations

- **Bias:** The model may reflect historical biases in care and documentation. Use of demographic features (age, gender, ethnicity) could perpetuate disparities if deployed without careful evaluation.
- **Transparency:** Model and SHAP outputs are interpretable, but users should not infer causation from feature contributions.
- **Accountability:** Predictions are probabilistic; they should never replace clinical judgment or informed consent.
- **Privacy:** Training data is de-identified per eICU terms; no patient-level data is stored by this application.

## Citation

If you use this model or code, please cite the eICU database:

> Pollard TJ, Johnson AEW, Raffa JD, Celi LA, Mark RG and Badawi O. *The eICU Collaborative Research Database, a freely available multi-center database for critical care research.* Scientific Data (2018). https://www.nature.com/articles/sdata2018178
