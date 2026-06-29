#!/usr/bin/env python3
"""
Detailed test for rule-based skill matching with fuzzy matching.
Demonstrates the Skill Match Score calculation and title relevance.
"""

import json
from app.models.candidate import Candidate
from app.models.jd import JobDescription
from app.services.relevance_evaluator import RelevanceEvaluator, RuleBasedSkillMatcher


def test_fuzzy_skill_matching():
    """Test fuzzy skill matching with Levenshtein distance."""
    
    print("\n" + "="*80)
    print("FUZZY SKILL MATCHING TEST")
    print("="*80)
    
    matcher = RuleBasedSkillMatcher()
    
    # Test cases: (candidate_skill, jd_skills, expected_match)
    test_cases = [
        ("Python", ["python", "java", "go"], True, "Exact match"),
        ("Python.js", ["Python", "JavaScript"], True, "Fuzzy match with typo"),
        ("React", ["React.js", "Vue", "Angular"], True, "Fuzzy match variation"),
        ("TensorFlow", ["Tensorflow", "PyTorch"], True, "Case-insensitive exact"),
        ("learning-to-rank", ["l2r"], True, "Fuzzy match with spacing"),
        ("Docker", ["Kubernetes", "AWS"], False, "No match"),
    ]
    
    print("\nTesting fuzzy matching algorithm:\n")
    for candidate_skill, jd_skills, expected_match, description in test_cases:
        is_match, score, matched_skill = matcher.find_skill_match(candidate_skill, jd_skills)
        status = "[PASS]" if is_match == expected_match else "[FAIL]"
        print(f"{status} | {description}")
        print(f"       Candidate: '{candidate_skill}' -> Matched: '{matched_skill}' (score: {score:.2f})")
        print()


def test_skill_match_score():
    """Test overall Skill Match Score calculation."""
    
    print("\n" + "="*80)
    print("SKILL MATCH SCORE CALCULATION TEST")
    print("="*80)
    
    matcher = RuleBasedSkillMatcher()
    
    # Sample JD required skills
    jd_required_skills = [
        "Python",
        "Machine Learning",
        "Vector Databases",
        "Embeddings",
        "Ranking Algorithms"
    ]
    
    # Test case 1: High skill match
    candidate_skills_1 = [
        "Python",
        "Machine Learning",
        "Milvus",  # fuzzy match for Vector Databases
        "Embeddings",
        "Ranking Models",  # fuzzy match for Ranking Algorithms
        "Kafka"
    ]
    
    score_1, matched_1, unmatched_1 = matcher.calculate_skill_match_score(
        candidate_skills_1, jd_required_skills
    )
    
    print(f"\nCandidate 1 (High Match):")
    print(f"  Skills: {candidate_skills_1}")
    print(f"  Match Score: {score_1:.1f}%")
    print(f"  Matched ({len(matched_1)}):")
    for m in matched_1:
        print(f"    - {m}")
    print(f"  Unmatched ({len(unmatched_1)}): {unmatched_1}")
    
    # Test case 2: Partial skill match
    candidate_skills_2 = [
        "Python",
        "Deep Learning",
        "Semantic Search",
        "NoSQL"
    ]
    
    score_2, matched_2, unmatched_2 = matcher.calculate_skill_match_score(
        candidate_skills_2, jd_required_skills
    )
    
    print(f"\nCandidate 2 (Partial Match):")
    print(f"  Skills: {candidate_skills_2}")
    print(f"  Match Score: {score_2:.1f}%")
    print(f"  Matched ({len(matched_2)}):")
    for m in matched_2:
        print(f"    - {m}")
    print(f"  Unmatched ({len(unmatched_2)}): {unmatched_2}")
    
    # Test case 3: Low skill match
    candidate_skills_3 = [
        "JavaScript",
        "React",
        "CSS",
        "HTML"
    ]
    
    score_3, matched_3, unmatched_3 = matcher.calculate_skill_match_score(
        candidate_skills_3, jd_required_skills
    )
    
    print(f"\nCandidate 3 (Low Match):")
    print(f"  Skills: {candidate_skills_3}")
    print(f"  Match Score: {score_3:.1f}%")
    print(f"  Matched ({len(matched_3)}):")
    for m in matched_3:
        print(f"    - {m}")
    print(f"  Unmatched ({len(unmatched_3)}): {unmatched_3}")


def test_experience_scoring():
    """Test experience scoring with penalties."""
    
    print("\n" + "="*80)
    print("EXPERIENCE REQUIREMENT MATCHING TEST")
    print("="*80)
    
    evaluator = RelevanceEvaluator()
    
    test_years = [1.5, 3.0, 5.0, 6.5, 9.0, 11.0, 14.0]
    
    print(f"\nExperience scoring (target range: 5-9 years):\n")
    
    for years in test_years:
        score, explanation = evaluator.calculate_experience_score(years)
        print(f"  {years:5.1f} years  >  Score: {score:+6.1f}  |  {explanation}")


def test_title_relevance():
    """Test job title relevance matching."""
    
    print("\n" + "="*80)
    print("JOB TITLE RELEVANCE TEST")
    print("="*80)
    
    evaluator = RelevanceEvaluator()
    
    jd_text = "Senior AI Engineer with strong embeddings experience"
    
    test_titles = [
        ("Senior AI Engineer", ["ML Engineer", "Data Scientist"]),
        ("Machine Learning Lead", ["Data Analyst", "BI Developer"]),
        ("Software Engineer II", ["Engineer", "Developer"]),
        ("AI Researcher", ["Research Associate", "PhD Student"]),
        ("Product Manager", ["Manager", "Coordinator"]),
    ]
    
    print(f"\nTitle relevance scoring:\n")
    
    for current_title, history_titles in test_titles:
        score, explanation = evaluator.calculate_title_relevance(
            current_title, history_titles, jd_text
        )
        print(f"  Current: '{current_title}'")
        print(f"    Score: {score:6.1f}  |  {explanation}")
        if history_titles:
            print(f"    History: {', '.join(history_titles)}")
        print()


def test_jd_skill_extraction():
    """Test extraction of required skills from JD text."""
    
    print("\n" + "="*80)
    print("JD SKILL EXTRACTION TEST")
    print("="*80)
    
    evaluator = RelevanceEvaluator()
    
    jd_text = """
    Senior AI Engineer — Founding Team
    
    Key Requirements:
    - Production experience with embeddings and retrieval systems
    - Strong experience with vector databases like Milvus and Weaviate
    - Advanced Python skills with NumPy and Pandas
    - Understanding of ranking evaluation frameworks (NDCG, MRR)
    - Distributed systems knowledge (Kafka, Spark)
    
    Nice-to-have:
    - Open-source contributions on GitHub
    - Learning-to-rank model experience
    - HR-tech or recruiting domain expertise
    """
    
    jd = JobDescription(raw_text=jd_text, paragraph_count=1)
    
    extracted_skills = evaluator.extract_jd_required_skills(jd)
    
    print(f"\nExtracted {len(extracted_skills)} skills from JD:\n")
    for i, skill in enumerate(extracted_skills, 1):
        print(f"  {i:2d}. {skill}")


if __name__ == "__main__":
    test_fuzzy_skill_matching()
    test_skill_match_score()
    test_experience_scoring()
    test_title_relevance()
    test_jd_skill_extraction()
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80 + "\n")
