"""Compare rules-only, ML-only, and hybrid routing on one held-out split.

This is an exploratory evaluation for a small synthetic dataset, not a real-world
performance estimate. The test split is never used to fit the classifier.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from routing.engine import rule_candidates, route_complaint  # noqa: E402

DATA_PATH = ROOT / "data" / "synthetic_grievance_dataset_v0_7.csv"
OUT_DIR = ROOT / "evaluation" / "results"
RANDOM_STATE = 42
TEST_SIZE = 0.25


def rules_prediction(text: str) -> tuple[str, bool, str]:
    """Predict only when rules identify exactly one category; otherwise review."""
    hits = rule_candidates(text)
    if len(hits) == 1:
        return next(iter(hits)), False, "one_rule_category_matched"
    if len(hits) > 1:
        return "needs_review", True, "multiple_rule_categories_matched"
    return "needs_review", True, "no_rule_match"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    required = {"complaint_text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {sorted(required)}")
    df = df.dropna(subset=["complaint_text", "label"]).copy()
    df["complaint_text"] = df["complaint_text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["complaint_text"] != "") & (df["label"] != "")].reset_index(drop=True)
    counts = df["label"].value_counts()
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError("Need at least two labels with at least two examples each.")

    indices = list(df.index)
    train_idx, test_idx = train_test_split(indices, test_size=TEST_SIZE, random_state=RANDOM_STATE,
                                           stratify=df["label"])
    train_df, test_df = df.loc[train_idx], df.loc[test_idx]
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    model.fit(train_df["complaint_text"], train_df["label"])
    labels = sorted(df["label"].unique().tolist())

    records = []
    for _, row in test_df.iterrows():
        text = row["complaint_text"]
        actual = row["label"]
        ml_pred = str(model.predict([text])[0])
        rule_pred, rule_review, rule_reason = rules_prediction(text)
        hybrid = route_complaint(model, text, threshold=0.55)
        records.append({
            "complaint_id": row.get("complaint_id", ""),
            "complaint_text": text,
            "actual_label": actual,
            "rules_prediction": rule_pred,
            "rules_review_required": rule_review,
            "rules_reason": rule_reason,
            "ml_prediction": ml_pred,
            "ml_review_required": False,
            "hybrid_prediction": str(hybrid.get("suggested_label") or "needs_review"),
            "hybrid_review_required": bool(hybrid.get("review_required", True)),
            "hybrid_reason_codes": ";".join(hybrid.get("reason_codes", [])),
            "hybrid_top_score_not_calibrated": hybrid.get("score"),
        })
    pred_df = pd.DataFrame(records)

    metrics_rows = []
    reports = {}
    for approach, pred_col, review_col in [
        ("rules_only", "rules_prediction", "rules_review_required"),
        ("ml_only", "ml_prediction", "ml_review_required"),
        ("hybrid", "hybrid_prediction", "hybrid_review_required"),
    ]:
        y_true = pred_df["actual_label"].tolist()
        y_pred = pred_df[pred_col].tolist()
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        reports[approach] = report
        metrics_rows.append({
            "approach": approach,
            "test_rows": len(pred_df),
            "accuracy_including_review_as_noncategory": accuracy_score(y_true, y_pred),
            "macro_f1_on_known_categories": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
            "macro_precision_on_known_categories": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
            "macro_recall_on_known_categories": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
            "review_rate": float(pred_df[review_col].mean()),
            "review_count": int(pred_df[review_col].sum()),
            "note": "Synthetic holdout only; review predictions count as incorrect for category metrics." if approach != "ml_only" else "Synthetic holdout only; ML-only has no review policy, so review rate is 0 by definition.",
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUT_DIR / "comparison_metrics.csv", index=False)
    pred_df.to_csv(OUT_DIR / "heldout_predictions.csv", index=False)
    (OUT_DIR / "per_class_reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    cm_rows = []
    for approach, pred_col in [("rules_only", "rules_prediction"), ("ml_only", "ml_prediction"), ("hybrid", "hybrid_prediction")]:
        cm = confusion_matrix(pred_df["actual_label"], pred_df[pred_col], labels=labels)
        for i, actual_label in enumerate(labels):
            for j, predicted_label in enumerate(labels):
                cm_rows.append({"approach": approach, "actual_label": actual_label,
                                "predicted_label": predicted_label, "count": int(cm[i, j])})
    pd.DataFrame(cm_rows).to_csv(OUT_DIR / "confusion_matrices.csv", index=False)

    print(f"Dataset rows: {len(df)} | Train rows: {len(train_df)} | Held-out rows: {len(test_df)}")
    print("Label counts (full synthetic dataset):")
    print(counts.sort_index().to_string())
    print("\nComparison metrics:")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved results to: {OUT_DIR}")
    print("Interpretation warning: these tiny synthetic holdout results are exploratory, not real-world evidence.")


if __name__ == "__main__":
    main()
