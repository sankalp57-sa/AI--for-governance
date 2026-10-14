"""Explainable hybrid grievance routing for a synthetic hackathon prototype.

All categories/rules are illustrative, not official department policy. Model
scores are not calibrated real-world confidence. A staff member must confirm
every final route; review_required indicates extra review is needed before that.
"""
from __future__ import annotations
import re
from typing import Any
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

RULES = {
    "sanitation": ["garbage", "rubbish", "waste", "trash", "dustbin", "waste bin", "kachra", "kooda"],
    "water_supply": ["no water", "water supply", "water shortage", "tap water", "paani nahi", "pani nahi"],
    "roads": ["pothole", "road", "pavement", "footpath", "sadak", "gaddha"],
    "street_lighting": ["streetlight", "street light", "street lamp", "light not working", "street lights", "streetlight not working"],
    "drainage": ["drain", "sewer", "blocked drain", "overflowing drain", "nali", "naali", "sewage"],
}
VAGUE_PHRASES = ["something is wrong", "please help", "nobody is fixing it", "not working", "problem near my house"]


def normalize_text(text: str) -> str:
    text = str(text or "").strip().lower()
    return re.sub(r"\s+", " ", text)


def rule_candidates(text: str) -> dict[str, list[str]]:
    normalized = normalize_text(text)
    hits: dict[str, list[str]] = {}
    for label, phrases in RULES.items():
        found = [phrase for phrase in phrases if phrase in normalized]
        if found:
            hits[label] = found
    return hits


def train_routing_model(df: pd.DataFrame, random_state: int = 42):
    required = {"complaint_text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {', '.join(sorted(missing))}")
    clean = df.dropna(subset=["complaint_text", "label"]).copy()
    clean["complaint_text"] = clean["complaint_text"].astype(str)
    clean = clean[clean["complaint_text"].str.strip().ne("")]
    if clean["label"].nunique() < 2 or clean["label"].value_counts().min() < 2:
        raise ValueError("Need at least two labels and at least two examples per label for a stratified split.")
    X_train, X_test, y_train, y_test = train_test_split(
        clean["complaint_text"], clean["label"], test_size=0.25,
        random_state=random_state, stratify=clean["label"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "test_rows": len(X_test),
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
        "report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        "y_test": y_test,
        "predictions": pred,
    }
    return model, metrics


def route_complaint(model: Pipeline, complaint: str, threshold: float = 0.55, margin_threshold: float = 0.10) -> dict[str, Any]:
    """Recommend a category and explicitly explain when additional review is needed."""
    original = str(complaint or "")
    normalized = normalize_text(original)
    if not normalized:
        return {
            "original_complaint": original, "suggested_label": None, "score": None,
            "second_label": None, "second_score": None, "review_required": True,
            "reason_codes": ["empty_input"], "rule_candidates": {}, "alternatives": [],
            "explanation": "No complaint text was provided. Ask the complainant to enter or provide the complaint.",
            "policy_note": "No route was recommended. Human confirmation is required for every final route.",
        }

    probs = model.predict_proba([original])[0]
    classes = list(model.named_steps["clf"].classes_)
    ranked = sorted(zip(classes, probs), key=lambda item: item[1], reverse=True)
    best_label, best_score = str(ranked[0][0]), float(ranked[0][1])
    second_label = str(ranked[1][0]) if len(ranked) > 1 else None
    second_score = float(ranked[1][1]) if len(ranked) > 1 else 0.0
    hits = rule_candidates(original)
    reasons: list[str] = []

    if best_score < threshold:
        reasons.append("below_review_threshold")
    if len(hits) > 1:
        reasons.append("multiple_rule_categories_matched")
    if hits and best_label not in hits:
        reasons.append("rules_and_model_disagree")
    if not hits:
        reasons.append("no_rule_evidence")
    if best_score - second_score < margin_threshold:
        reasons.append("top_two_model_scores_close")
    if any(phrase in normalized for phrase in VAGUE_PHRASES):
        reasons.append("vague_complaint_text")
    if best_label == "needs_review":
        reasons.append("model_predicted_needs_review")

    # A clear, single rule match is supporting evidence, not a guarantee.
    if len(hits) == 1:
        rule_label = next(iter(hits))
        if rule_label == best_label:
            explanation = f"The model suggestion matches the rule evidence for {rule_label.replace('_', ' ')}. Matched phrase(s): {', '.join(hits[rule_label])}."
        else:
            explanation = f"Rule evidence points to {rule_label.replace('_', ' ')}, while the model suggests {best_label.replace('_', ' ')}; compare both before deciding."
    elif len(hits) > 1:
        explanation = "The complaint contains phrases associated with multiple departments: " + ", ".join(label.replace('_', ' ') for label in hits) + ". The primary issue is ambiguous."
    else:
        explanation = "No configured rule phrase matched. The model suggestion is based on learned patterns in a small synthetic dataset, so treat it cautiously."

    return {
        "original_complaint": original,
        "suggested_label": best_label,
        "score": best_score,
        "second_label": second_label,
        "second_score": second_score,
        "review_required": bool(reasons),
        "reason_codes": reasons or ["no_extra_review_triggered_by_demo_policy"],
        "rule_candidates": hits,
        "alternatives": [{"label": str(label), "score": float(score)} for label, score in ranked[:3]],
        "explanation": explanation,
        "policy_note": "Recommendation only. A staff member must confirm the final department even when no extra review trigger fires.",
    }
