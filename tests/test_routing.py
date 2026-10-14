from pathlib import Path
import pandas as pd
from routing.engine import train_routing_model, route_complaint, rule_candidates

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic_grievance_dataset_v0_7.csv"

def get_model():
    return train_routing_model(pd.read_csv(DATA, encoding="utf-8-sig"))[0]

def test_empty_input_requests_review():
    result = route_complaint(get_model(), "   ")
    assert result["review_required"] is True
    assert "empty_input" in result["reason_codes"]
    assert result["suggested_label"] is None

def test_water_rule_matches():
    assert "water_supply" in rule_candidates("There is no water supply today")

def test_cross_department_complaint_flags_multiple_rules():
    result = route_complaint(get_model(), "The drain is blocked and the road is flooded")
    assert result["review_required"] is True
    assert "multiple_rule_categories_matched" in result["reason_codes"]
    assert "ambiguous" in result["explanation"].lower()

def test_output_has_human_review_fields():
    result = route_complaint(get_model(), "Garbage has not been collected for a week")
    assert "suggested_label" in result
    assert "alternatives" in result
    assert "review_required" in result
    assert "explanation" in result

def test_no_rule_match_requires_review():
    result = route_complaint(get_model(), "The issue is affecting everyone in my area")
    assert result["review_required"] is True
    assert "no_rule_evidence" in result["reason_codes"]

def test_rule_model_disagreement_requires_review():
    model = get_model()
    result = route_complaint(model, "Garbage collection and water supply are both affected")
    assert result["review_required"] is True
    assert "multiple_rule_categories_matched" in result["reason_codes"]

def test_vague_complaint_requires_review():
    result = route_complaint(get_model(), "Something is wrong near my house, please help")
    assert result["review_required"] is True
    assert "vague_complaint_text" in result["reason_codes"]

def test_hindi_water_phrase_matches_rule():
    assert "water_supply" in rule_candidates("Hamare yahan paani nahi aa raha")

def test_hindi_sanitation_phrase_matches_rule():
    assert "sanitation" in rule_candidates("Hamari gali mein kachra jama ho gaya hai")

def test_blank_or_whitespace_is_not_routed():
    result = route_complaint(get_model(), "   \n  ")
    assert result["suggested_label"] is None
    assert result["review_required"] is True
