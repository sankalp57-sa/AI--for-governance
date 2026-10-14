"""Audit dataset quality before training. Does not change or relabel data."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic_grievance_dataset_v0_7.csv"
OUT = ROOT / "evaluation" / "results"


def main():
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    required = {"complaint_text", "label"}
    missing_columns = sorted(required - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df["complaint_text"] = df["complaint_text"].fillna("").astype(str).str.strip()
    df["label"] = df["label"].fillna("").astype(str).str.strip()
    df["normalized_text"] = df["complaint_text"].str.lower().str.replace(r"\s+", " ", regex=True)

    exact_dupes = df[df["normalized_text"].duplicated(keep=False)].sort_values("normalized_text")
    conflicting = (df.groupby("normalized_text")["label"].nunique().reset_index(name="label_count"))
    conflicting_texts = set(conflicting.loc[conflicting["label_count"] > 1, "normalized_text"])
    conflict_rows = df[df["normalized_text"].isin(conflicting_texts)].sort_values("normalized_text")

    summary = {
        "dataset_file": DATA.name,
        "rows": int(len(df)),
        "empty_complaint_text_rows": int(df["complaint_text"].eq("").sum()),
        "empty_label_rows": int(df["label"].eq("").sum()),
        "duplicate_normalized_text_groups": int(exact_dupes["normalized_text"].nunique()),
        "rows_in_duplicate_normalized_text_groups": int(len(exact_dupes)),
        "conflicting_label_text_groups": int(len(conflicting_texts)),
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts(dropna=False).items()},
        "language_group_counts": {str(k): int(v) for k, v in df["language_group"].value_counts(dropna=False).items()} if "language_group" in df.columns else {},
        "warning": "This script detects basic data-quality issues only. It cannot establish that labels are correct or examples represent real complaints. No rows are automatically removed or relabeled."
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dataset_audit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    exact_dupes.drop(columns=["normalized_text"]).to_csv(OUT / "duplicate_text_rows_for_review.csv", index=False)
    conflict_rows.drop(columns=["normalized_text"]).to_csv(OUT / "conflicting_label_rows_for_review.csv", index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nAudit outputs saved to: {OUT}")

if __name__ == "__main__":
    main()
