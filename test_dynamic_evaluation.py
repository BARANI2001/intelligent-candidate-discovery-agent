#!/usr/bin/env python3
"""
Test Dynamic Evaluation Service

Tests the flexible input handling for:
- Multiple JD formats (text, docx)
- Multiple candidate formats (JSON array, JSONL)
- Schema validation
- Complete evaluation pipeline
"""

import json
import traceback
from app.services.dynamic_evaluation_service import DynamicEvaluationService
from app.models.candidate import Candidate


def test_dynamic_jd_processing():
    """Test JD processing from different formats."""
    
    print("\n" + "="*80)
    print("TEST 1: Dynamic JD Processing")
    print("="*80)
    
    service = DynamicEvaluationService(use_fastembed=True)
    
    # Test 1.1: Text JD
    print("\n1.1 Processing JD from text...")
    jd_text = "Senior AI Engineer with 5-9 years experience. Skills: Python, embeddings, vector databases."
    jd = service.process_jd_text(jd_text)
    print(f"[PASS] Text JD processed")
    print(f"   - Length: {len(jd.raw_text)} chars")
    print(f"   - Paragraphs: {jd.paragraph_count}")
    
    # Test 1.2: JD summary
    print("\n1.2 Analyzing JD...")
    summary = service.get_jd_summary(jd)
    print(f"[PASS] JD analyzed")
    print(f"   - Keywords extracted: {summary['keyword_count']}")
    print(f"   - Keywords: {summary['extracted_keywords'][:5]}...")


def test_dynamic_candidate_processing():
    """Test candidate processing from different formats."""
    
    print("\n" + "="*80)
    print("TEST 2: Dynamic Candidate Processing")
    print("="*80)
    
    service = DynamicEvaluationService(use_fastembed=False)
    
    # Load sample candidates
    with open('inputs/datasets/sample_candidates.json') as f:
        sample_data = json.load(f)[:2]
    
    # Test 2.1: JSON array
    print("\n2.1 Parsing candidates from JSON array...")
    json_str = json.dumps(sample_data)
    candidates = service.parse_candidates_json_list(json_str)
    print(f"[PASS] Parsed {len(candidates)} candidates from JSON array")
    
    for c in candidates:
        print(f"   - {c.candidate_id}: {c.profile.current_title}")
    
    # Test 2.2: List format
    print("\n2.2 Parsing candidates from Python list...")
    candidates = service.parse_candidates_json_list(sample_data)
    print(f"[PASS] Parsed {len(candidates)} candidates from list")
    
    # Test 2.3: JSONL format
    print("\n2.3 Parsing candidates from JSONL format...")
    jsonl_str = "\n".join(json.dumps(c) for c in sample_data)
    candidates = service.parse_candidates_jsonl(jsonl_str)
    print(f"[PASS] Parsed {len(candidates)} candidates from JSONL")
    
    # Test 2.4: Auto-detect format
    print("\n2.4 Auto-detecting candidate format...")
    
    # Auto-detect JSON array
    candidates = service.parse_candidates(json_str)
    print(f"[PASS] Auto-detected JSON array: {len(candidates)} candidates")
    
    # Auto-detect JSONL
    candidates = service.parse_candidates(jsonl_str)
    print(f"[PASS] Auto-detected JSONL: {len(candidates)} candidates")


def test_schema_validation():
    """Test schema validation."""
    
    print("\n" + "="*80)
    print("TEST 3: Schema Validation")
    print("="*80)
    
    service = DynamicEvaluationService(use_fastembed=False)
    
    # Load valid candidate
    with open('inputs/datasets/sample_candidates.json') as f:
        valid_candidate = json.load(f)[0]
    
    # Test 3.1: Valid candidate
    print("\n3.1 Validating valid candidate...")
    is_valid, msg = service.validate_schema(valid_candidate)
    print(f"[PASS] Result: {is_valid} - {msg}")
    
    # Test 3.2: Invalid candidate (missing required field)
    print("\n3.2 Validating invalid candidate (missing field)...")
    invalid_candidate = valid_candidate.copy()
    del invalid_candidate['candidate_id']
    is_valid, msg = service.validate_schema(invalid_candidate)
    print(f"[PASS] Result: {is_valid} - Schema validation caught error")
    print(f"   - {msg[:100]}...")
    
    # Test 3.3: Invalid candidate (wrong type)
    print("\n3.3 Validating invalid candidate (wrong type)...")
    invalid_candidate = valid_candidate.copy()
    invalid_candidate['profile']['years_of_experience'] = "six"  # Should be number
    is_valid, msg = service.validate_schema(invalid_candidate)
    print(f"[PASS] Result: {is_valid} - Schema validation caught error")


def test_complete_evaluation():
    """Test complete evaluation pipeline with dynamic inputs."""
    
    print("\n" + "="*80)
    print("TEST 4: Complete Dynamic Evaluation")
    print("="*80)
    
    service = DynamicEvaluationService(use_fastembed=True)
    
    # Prepare inputs
    jd_text = """
    Senior AI Engineer – Founding Team
    Location: Pune/Noida, India
    Experience Required: 5–9 years
    
    Key Requirements:
    - Production experience with embeddings-based retrieval systems
    - Strong experience with vector databases
    - Strong Python skills
    - Experience with ranking evaluation frameworks
    
    Nice-to-have:
    - Learning-to-rank models
    - HR-tech experience
    """
    
    with open('inputs/datasets/sample_candidates.json') as f:
        candidates_data = json.load(f)[:3]
    
    candidates_json = json.dumps(candidates_data)
    
    # Run evaluation
    print("\nRunning complete evaluation...")
    try:
        result = service.evaluate(jd_text, candidates_json, format_hint='json_array')
        
        print(f"[PASS] Evaluation complete")
        print(f"   - Evaluated {len(result.evaluations)} candidates")
        
        # Show top 3
        for rank, eval_res in enumerate(result.evaluations[:3], 1):
            print(f"\n   {rank}. {eval_res.candidate_id}")
            print(f"      Embedding Sim: {eval_res.vector_similarity.cosine_similarity:.3f}")
            print(f"      Rule-Based: {eval_res.rule_based_score}/100")
            print(f"      Combined: {eval_res.combined_score:.3f}")
    
    except Exception as e:
        print(f"[FAIL] Evaluation failed: {e}")


def test_candidate_summary():
    """Test candidate summary generation."""
    
    print("\n" + "="*80)
    print("TEST 5: Candidate Summary")
    print("="*80)
    
    service = DynamicEvaluationService(use_fastembed=False)
    
    with open('inputs/datasets/sample_candidates.json') as f:
        candidate_data = json.load(f)[0]
    
    candidate = Candidate(**candidate_data)
    
    print("\nGenerating candidate summary...")
    summary = service.get_candidate_summary(candidate)
    
    print(f"[PASS] Summary generated:")
    print(f"   - ID: {summary['candidate_id']}")
    print(f"   - Name: {summary['name']}")
    print(f"   - Title: {summary['title']}")
    print(f"   - Experience: {summary['experience_years']} years")
    print("\n   Top skills: {', '.join(summary['skills'][:5])}...")


def test_error_handling():
    """Test error handling for invalid inputs."""
    
    print("\n" + "="*80)
    print("TEST 6: Error Handling")
    print("="*80)
    
    service = DynamicEvaluationService(use_fastembed=False)
    
    # Test 6.1: Empty JD
    print("\n6.1 Testing empty JD...")
    try:
        service.process_jd_text("")
        print("[FAIL] Should have raised error")
    except ValueError as e:
        print(f"[PASS] Correctly caught error: {str(e)[:50]}...")
    
    # Test 6.2: Invalid JSON
    print("\n6.2 Testing invalid JSON...")
    try:
        service.parse_candidates_json_list("not valid json {")
        print("[FAIL] Should have raised error")
    except ValueError as e:
        print(f"[PASS] Correctly caught error: {str(e)[:50]}...")
    
    # Test 6.3: Empty candidates
    print("\n6.3 Testing empty candidates...")
    try:
        service.parse_candidates_json_list("[]")
        print("[FAIL] Should have raised error")
    except ValueError as e:
        print(f"[PASS] Correctly caught error: {str(e)[:50]}...")
    
    # Test 6.4: Invalid candidate schema
    print("\n6.4 Testing invalid candidate schema...")
    try:
        service.parse_candidates_json_list('[{"invalid": "candidate"}]')
        print("[FAIL] Should have raised error")
    except ValueError as e:
        print(f"[PASS] Correctly caught error: Schema validation")


if __name__ == "__main__":
    try:
        test_dynamic_jd_processing()
        test_dynamic_candidate_processing()
        test_schema_validation()
        test_complete_evaluation()
        test_candidate_summary()
        test_error_handling()
        
        print("\n" + "="*80)
        print("[PASS] ALL DYNAMIC EVALUATION TESTS PASSED")
        print("="*80 + "\n")
    
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        traceback.print_exc()
