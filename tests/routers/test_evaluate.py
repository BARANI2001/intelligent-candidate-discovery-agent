
import io
import json

import pytest
from docx import Document
from fastapi.testclient import TestClient

from app.main import app
from app.services.docx_parser import extract_jd_text

client = TestClient(app)


def make_docx_bytes(paragraphs):
    doc = Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def make_valid_candidate(candidate_id="CAND_0000001"):
    """
    Returns a full, schema-valid candidate dict. Tests that want to check
    a validation failure should copy this and break exactly one field,
    so it's clear what's actually being tested.
    """
    return {
        "candidate_id": candidate_id,
        "profile": {
            "anonymized_name": "Test Candidate",
            "headline": "Senior ML Engineer",
            "summary": "Built ranking and retrieval systems at a product company.",
            "location": "Pune",
            "country": "India",
            "years_of_experience": 6.0,
            "current_title": "Senior ML Engineer",
            "current_company": "TestCo",
            "current_company_size": "201-500",
            "current_industry": "AI/ML",
        },
        "career_history": [
            {
                "company": "TestCo",
                "title": "Senior ML Engineer",
                "start_date": "2021-01-01",
                "end_date": None,
                "duration_months": 36,
                "is_current": True,
                "industry": "AI/ML",
                "company_size": "201-500",
                "description": "Built embeddings-based retrieval system in production.",
            }
        ],
        "education": [
            {
                "institution": "IIT Bombay",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": 2012,
                "end_year": 2016,
                "grade": "8.7",
                "tier": "tier_1",
            }
        ],
        "skills": [
            {"name": "Python", "proficiency": "expert", "endorsements": 12, "duration_months": 72}
        ],
        "certifications": [],
        "languages": [{"language": "English", "proficiency": "professional"}],
        "redrob_signals": {
            "profile_completeness_score": 85.0,
            "signup_date": "2022-01-01",
            "last_active_date": "2026-06-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 20,
            "applications_submitted_30d": 2,
            "recruiter_response_rate": 0.7,
            "avg_response_time_hours": 5.0,
            "skill_assessment_scores": {"python": 90.0},
            "connection_count": 300,
            "endorsements_received": 15,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 30.0, "max": 40.0},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 65.0,
            "search_appearance_30d": 10,
            "saved_by_recruiters_30d": 3,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.8,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }


def post_evaluate(jd_paragraphs, candidates, filename="jd.docx"):
    """Helper: builds the multipart request the endpoint expects."""
    jd_bytes = make_docx_bytes(jd_paragraphs)
    return client.post(
        "/evaluate-candidates",
        files={
            "jd_file": (
                filename,
                jd_bytes,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
        data={"candidates_json": json.dumps(candidates)},
    )

# --- docx_parser.py tests (JD extraction) ---

def test_extract_jd_text_returns_paragraphs():
    file_bytes = make_docx_bytes(["Title", "Body line one.", "Body line two."])
    result = extract_jd_text(file_bytes)
    assert result.paragraph_count == 3
    assert "Title" in result.raw_text
    assert "Body line two." in result.raw_text


def test_extract_jd_text_rejects_garbage_bytes():
    with pytest.raises(Exception):
        extract_jd_text(b"this is not a docx file")


def test_extract_jd_text_rejects_empty_doc():
    file_bytes = make_docx_bytes([])
    with pytest.raises(Exception):
        extract_jd_text(file_bytes)


def test_extract_jd_text_ignores_blank_paragraphs():
    # Blank paragraphs (e.g. spacing in the original doc) shouldn't count
    # toward paragraph_count or pollute raw_text.
    file_bytes = make_docx_bytes(["Real content.", "", "   ", "More content."])
    result = extract_jd_text(file_bytes)
    assert result.paragraph_count == 2


# --- /evaluate-candidates: happy path ---

def test_evaluate_candidates_happy_path():
    response = post_evaluate(
        ["Senior AI Engineer JD text."],
        [make_valid_candidate()],
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job_description"]["paragraph_count"] == 1
    assert len(body["candidates"]) == 1

    candidate = body["candidates"][0]
    assert candidate["candidate_id"] == "CAND_0000001"
    assert "redrob_signals" in candidate
    assert candidate["redrob_signals"]["expected_salary_range_inr_lpa"] == {
        "min": 30.0,
        "max": 40.0,
    }


def test_evaluate_candidates_accepts_multiple_candidates():
    candidates = [
        make_valid_candidate("CAND_0000001"),
        make_valid_candidate("CAND_0000002"),
    ]
    response = post_evaluate(["JD text."], candidates)
    assert response.status_code == 200
    body = response.json()
    assert len(body["candidates"]) == 2
    ids = {c["candidate_id"] for c in body["candidates"]}
    assert ids == {"CAND_0000001", "CAND_0000002"}


# --- /evaluate-candidates: JD file validation ---

def test_evaluate_candidates_rejects_non_docx_extension():
    response = post_evaluate(["JD text."], [make_valid_candidate()], filename="jd.txt")
    assert response.status_code == 400
    assert "docx" in response.json()["detail"].lower()


def test_evaluate_candidates_rejects_empty_jd():
    response = post_evaluate([], [make_valid_candidate()])
    assert response.status_code == 400


# --- /evaluate-candidates: candidates_json parsing ---

def test_evaluate_candidates_rejects_invalid_json():
    jd_bytes = make_docx_bytes(["JD text."])
    response = client.post(
        "/evaluate-candidates",
        files={
            "jd_file": (
                "jd.docx",
                jd_bytes,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
        data={"candidates_json": "not json"},
    )
    assert response.status_code == 400


def test_evaluate_candidates_rejects_non_list_json():
    # A single object instead of a JSON array should be rejected, not
    # silently wrapped or partially processed.
    response = post_evaluate(["JD text."], make_valid_candidate())  # dict, not list
    assert response.status_code == 400


def test_evaluate_candidates_rejects_empty_candidate_list():
    response = post_evaluate(["JD text."], [])
    assert response.status_code == 400


# --- /evaluate-candidates: candidate schema validation ---

def test_evaluate_candidates_rejects_missing_required_fields():
    response = post_evaluate(["JD text."], [{"candidate_id": "CAND_0000001"}])
    assert response.status_code == 422


def test_evaluate_candidates_rejects_missing_redrob_signals():
    bad = make_valid_candidate()
    del bad["redrob_signals"]
    response = post_evaluate(["JD text."], [bad])
    assert response.status_code == 422


def test_evaluate_candidates_rejects_malformed_candidate_id():
    bad = make_valid_candidate(candidate_id="not-a-valid-id")
    response = post_evaluate(["JD text."], [bad])
    assert response.status_code == 422


def test_evaluate_candidates_rejects_lowercase_candidate_id_prefix():
    bad = make_valid_candidate(candidate_id="cand_0000001")
    response = post_evaluate(["JD text."], [bad])
    assert response.status_code == 422


def test_evaluate_candidates_rejects_empty_career_history():
    bad = make_valid_candidate()
    bad["career_history"] = []
    response = post_evaluate(["JD text."], [bad])
    assert response.status_code == 422


def test_evaluate_candidates_rejects_invalid_education_tier():
    bad = make_valid_candidate()
    bad["education"][0]["tier"] = "Tier 1"  # schema wants "tier_1"
    response = post_evaluate(["JD text."], [bad])
    assert response.status_code == 422


def test_evaluate_candidates_rejects_invalid_skill_proficiency():
    bad = make_valid_candidate()
    bad["skills"][0]["proficiency"] = "godlike"
    response = post_evaluate(["JD text."], [bad])
    assert response.status_code == 422


def test_evaluate_candidates_rejects_invalid_company_size():
    bad = make_valid_candidate()
    bad["profile"]["current_company_size"] = "huge"
    response = post_evaluate(["JD text."], [bad])
    assert response.status_code == 422


def test_evaluate_candidates_rejects_invalid_work_mode():
    bad = make_valid_candidate()
    bad["redrob_signals"]["preferred_work_mode"] = "space"
    response = post_evaluate(["JD text."], [bad])
    assert response.status_code == 422


def test_evaluate_candidates_rejects_out_of_range_response_rate():
    bad = make_valid_candidate()
    bad["redrob_signals"]["recruiter_response_rate"] = 1.5  # must be 0.0-1.0
    response = post_evaluate(["JD text."], [bad])
    assert response.status_code == 422


def test_evaluate_candidates_accepts_no_github_sentinel():
    # -1 is the documented sentinel for "no GitHub linked" -- must be
    # accepted, not rejected as out-of-range.
    ok = make_valid_candidate()
    ok["redrob_signals"]["github_activity_score"] = -1
    response = post_evaluate(["JD text."], [ok])
    assert response.status_code == 200


def test_evaluate_candidates_rejects_one_valid_one_invalid_in_batch():
    # If any candidate in the batch fails validation, the whole request
    # should fail -- partial success would silently drop a candidate
    # from the dataset without telling the caller.
    good = make_valid_candidate("CAND_0000001")
    bad = make_valid_candidate("CAND_0000002")
    bad["career_history"] = []
    response = post_evaluate(["JD text."], [good, bad])
    assert response.status_code == 422
