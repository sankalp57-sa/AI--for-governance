# Explainable Grievance Routing Assistant — v0.7

A hackathon prototype for exploring grievance routing using illustrative rules, TF-IDF + Logistic Regression, a review policy, an audit log and evaluation scripts. All examples are synthetic. This is not an official government routing system.

## Run on Windows PowerShell

From the folder containing `app.py`:

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Run automated tests:

```powershell
py -m pytest -q
```

Run model comparison:

```powershell
py evaluation/evaluate.py
```

Audit dataset quality without changing labels or deleting rows:

```powershell
py evaluation/audit_dataset.py
```

## v0.7 changes

- Added a dataset audit for empty fields, normalized exact duplicate text, conflicting labels, label counts and language-group counts.
- Audit creates `dataset_audit.json`, `duplicate_text_rows_for_review.csv`, and `conflicting_label_rows_for_review.csv` in `evaluation/results/`.
- Audit does not automatically delete, relabel, or alter dataset records.
- Added routing rule tests for Hindi/Hinglish phrase examples and blank-input safety.
- Preserved existing Streamlit workflow and rules/ML/hybrid evaluation outputs.

## Important interpretation notes

- Dataset quality checks do not prove labels are correct or representative.
- The dataset is small, synthetic, and hand-authored; held-out scores are exploratory only.
- A Logistic Regression probability is not calibrated real-world confidence.
- Categories and phrase rules are illustrative, not official policy.
- Human confirmation is required for every final department assignment. The app does not submit cases to a municipality.
- Local CSV audit storage is demo-only and lacks production authentication, access controls, retention policy, and privacy safeguards.
