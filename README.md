# One Complaint, Five Departments
## AI-Assisted Grievance Routing for Governance

An explainable, human-in-the-loop prototype for routing citizen complaints to the appropriate government department.

**Hackathon:** AI Synergy Hackathon 2026 â€” Governance
**Version:** v0.8
**Repository:** https://github.com/sankalp57-sa/AI--for-governance

## Problem Statement

A citizen complaint may describe multiple civic issues, use vague wording, or contain multilingual text. Incorrect routing can delay resolution. This project explores how rule-based matching and machine learning can support routing recommendations while preserving human oversight.

## Features

- Rule-based complaint matching using category-related phrases.
- TF-IDF and logistic-regression classification baseline.
- Hybrid routing recommendations and review flags.
- Explanation, matched rule evidence, reason codes, and alternative categories.
- Dataset quality audit for missing values, duplicate text, and conflicting labels.
- Comparative evaluation of rules-only, ML-only, and hybrid approaches.
- Human confirmation required for final routing decisions.

## Supported Categories

| Category | Example issue |
|---|---|
| `sanitation` | Waste collection and cleanliness |
| `water_supply` | Water availability and supply |
| `roads` | Road damage and potholes |
| `street_lighting` | Streetlights not working |
| `drainage` | Blocked drains and drainage issues |

## How It Works

1. Accept complaint text.
2. Normalize text for consistent matching.
3. Identify category-related phrases and retain matching evidence.
4. Generate a machine-learning category prediction and score.
5. Combine available evidence to recommend a route or request review.
6. Have a human review and confirm the final routing decision.

**Important:** Model scores are not calibrated real-world probabilities. The system recommends routes; it does not independently resolve complaints.

## Evaluation

The evaluation scripts compare:

- Rules-only routing
- ML-only routing
- Hybrid routing

Metrics include accuracy, macro-averaged precision, recall and F1, review rate, and per-class results. The dataset audit checks basic data-quality issues.

Results must be reported from the actual evaluation output. Synthetic-data results are exploratory and do not demonstrate real-world accuracy, fairness, or generalizability.

## Running the Project

Use the project's existing Python environment and install dependencies from its dependency file.

Run tests:

```powershell
python -m pytest
```

Run the dataset audit:

```powershell
python .\evaluation\audit_dataset.py
```

Run comparative evaluation:

```powershell
python .\evaluation\evaluate.py
```

If the Streamlit entry point is `app.py`, launch it with:

```powershell
streamlit run app.py
```

Check the repository for the actual application entry point if the filename differs.

## Dataset and Limitations

The current evaluation dataset is synthetic. It must not be represented as real government grievance data.

- Routing recommendations require human confirmation.
- Model scores are not calibrated confidence probabilities.
- Duplicate text does not automatically mean a record is invalid.
- Synthetic evaluation does not prove real-world performance.
- Language-group fairness, accessibility, privacy, security, and operational readiness need further validation.
- This prototype is not a production-ready government grievance system.

## Version 1 Scope

The prototype focuses on complaint categorization, routing recommendations, explanations, review flags, and exploratory evaluation.

Live government-system integrations, automatic complaint resolution, and verified real-world performance claims are not established by this prototype.

## Version 0.8

Version 0.8 improves rule matching through whole-word matching and flexible whitespace handling.

## License

No license is specified here. Add the applicable license if one has been selected for this repository.
