import csv
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from routing.engine import train_routing_model, route_complaint

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "synthetic_grievance_dataset_v0_7.csv"
AUDIT_PATH = ROOT / "review_audit_log.csv"

st.set_page_config(page_title="Explainable Grievance Routing Assistant", page_icon="🏛️", layout="wide")
st.title("🏛️ Explainable Grievance Routing Assistant")
st.caption("Hackathon prototype v0.7 · Synthetic data only · Not for real municipal use")
st.warning("All examples are synthetic. Metrics are internal demonstrations, not evidence of real-world accuracy. Human confirmation is required before final routing.")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, encoding="utf-8-sig")

@st.cache_resource
def get_model():
    return train_routing_model(load_data())

df = load_data()
model, metrics = get_model()
with st.sidebar:
    st.header("Routing policy")
    threshold = st.slider("Demo review threshold", 0.20, 0.95, 0.55, 0.05)
    st.caption("Illustrative setting only. The score is not calibrated confidence.")

tab1, tab2, tab3, tab4 = st.tabs(["Try a complaint", "Evaluation snapshot", "Audit log", "Possible duplicates"])

with tab1:
    complaint = st.text_area("Complaint text", placeholder="Example: Nali ka paani sadak par bhar raha hai.")
    channel = st.selectbox("Source channel (demo)", ["Portal", "Phone", "Social media", "In-person", "Other"])
    if st.button("Analyse complaint", type="primary"):
        if not complaint.strip():
            st.error("Enter a complaint first.")
        else:
            st.session_state["last_result"] = route_complaint(model, complaint, threshold)
            st.session_state["last_channel"] = channel
    result = st.session_state.get("last_result")
    if result:
        st.subheader("Routing recommendation")
        if result["suggested_label"] is None:
            st.error("No usable complaint text. Enter a complaint to continue.")
        else:
            c1, c2 = st.columns(2)
            c1.metric("Suggested department", result["suggested_label"].replace("_", " ").title())
            c2.metric("Top model score (not calibrated confidence)", f"{result['score']:.3f}")
            if result["review_required"]:
                st.error("Human review required before routing.")
            else:
                st.success("No review trigger fired under the demo policy. Staff confirmation is still required.")
            st.write("**Why was this suggested?**")
            st.write(result.get("explanation", "No explanation available."))
            st.write("**Review reason codes**")
            st.write(", ".join(result["reason_codes"]).replace("_", " ").capitalize())
            st.write("**Rule-based evidence**")
            if result["rule_candidates"]:
                for label, phrases in result["rule_candidates"].items():
                    st.write(f"- {label.replace('_', ' ').title()}: matched phrase(s) {', '.join(phrases)}")
            else:
                st.write("No configured rule phrase matched. This does not mean the complaint is invalid.")
            st.write("**Alternative model suggestions**")
            st.dataframe(pd.DataFrame(result["alternatives"]), use_container_width=True, hide_index=True)
            st.caption(result["policy_note"])
            labels = sorted(model.named_steps["clf"].classes_)
            default_label = result["suggested_label"] if result["suggested_label"] in labels else labels[0]
            chosen = st.selectbox("Staff-confirmed label", labels, index=labels.index(default_label))
            reviewer = st.text_input("Reviewer demo ID (do not enter real personal identifiers)", value="demo-reviewer-1")
            note = st.text_input("Review note / reason for correction (optional)")
            if st.button("Save human decision"):
                row = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "complaint_text": result.get("original_complaint", ""),
                    "source_channel": st.session_state.get("last_channel", "Other"),
                    "model_suggestion": result.get("suggested_label", ""),
                    "top_model_score_not_calibrated": result.get("score", ""),
                    "review_required": result.get("review_required", True),
                    "reason_codes": ";".join(result.get("reason_codes", [])),
                    "rule_candidates": repr(result.get("rule_candidates", {})),
                    "human_decision": chosen,
                    "reviewer_demo_id": reviewer,
                    "review_note": note,
                    "model_version": "v0.7-hybrid-tfidf-logreg",
                    "data_origin": "synthetic_demo",
                }
                exists = AUDIT_PATH.exists()
                with AUDIT_PATH.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if not exists:
                        writer.writeheader()
                    writer.writerow(row)
                st.success("Human decision saved to the local demo audit log.")
                st.caption("This does not submit a complaint to any municipality.")

with tab2:
    c1, c2, c3 = st.columns(3)
    c1.metric("Held-out test rows", metrics["test_rows"])
    c2.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    c3.metric("Macro-F1", f"{metrics['macro_f1']:.3f}")
    st.caption("Random stratified split of the small synthetic dataset. These results are not a real-world performance estimate.")
    report_df = pd.DataFrame(metrics["report"]).T
    st.dataframe(report_df, use_container_width=True)
    st.write("Dataset label counts")
    st.bar_chart(df["label"].value_counts())

with tab3:
    if AUDIT_PATH.exists():
        audit = pd.read_csv(AUDIT_PATH)
        st.dataframe(audit, use_container_width=True)
        st.download_button("Download audit CSV", audit.to_csv(index=False), "review_audit_log.csv", "text/csv")
    else:
        st.info("No review decisions saved yet. Analyse a complaint and save a staff decision in the first tab.")

with tab4:
    st.subheader("Possible duplicate complaint candidates")
    st.caption("Similarity compares against synthetic examples only. It is a review aid, not proof of duplication.")
    query = st.text_area("Complaint to compare", key="duplicate_query", placeholder="Enter complaint text")
    sim_threshold = st.slider("Similarity threshold", 0.10, 0.90, 0.35, 0.05, key="sim_threshold")
    if st.button("Find similar examples"):
        if not query.strip():
            st.error("Enter a complaint to compare.")
        else:
            corpus = df["complaint_text"].astype(str).tolist()
            vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1)
            matrix = vectorizer.fit_transform(corpus + [query])
            sims = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
            results = pd.DataFrame({"complaint_id": df["complaint_id"], "candidate_text": df["complaint_text"], "synthetic_label": df["label"], "language_group": df["language_group"], "similarity": sims}).sort_values("similarity", ascending=False)
            candidates = results[results["similarity"] >= sim_threshold].head(5)
            if candidates.empty:
                st.info("No candidates exceeded this threshold. This does not prove no duplicate exists.")
            else:
                st.dataframe(candidates, use_container_width=True, hide_index=True)
                st.warning("Human review required. Do not automatically merge or close complaints.")

st.divider()
st.caption("Limitations: small hand-authored synthetic dataset; illustrative categories; no authentication or production privacy controls; keyword rules are incomplete; scores may be poorly calibrated; staff review is mandatory.")
