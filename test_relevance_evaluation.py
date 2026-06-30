#!/usr/bin/env python3
"""
Test script for the relevance evaluation system.
Demonstrates vector similarity and rule-based candidate scoring.
"""

import json
from app.models.candidate import Candidate
from app.models.jd import JobDescription
from app.services.relevance_evaluator import RelevanceEvaluator


def test_relevance_evaluation():
    """Test the relevance evaluator with sample data."""
    
    # Create a sample JD
    jd_text = """
    Senior AI Engineer — Founding Team
    Location: Pune/Noida, India
    Experience Required: 5–9 years
    
    Key Requirements:
    - Production experience with embeddings-based retrieval systems
    - Strong experience with vector databases (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch)
    - Strong Python skills
    - Experience designing evaluation frameworks for ranking systems (NDCG, MRR, MAP)
    - Understanding of hybrid retrieval vs dense retrieval
    
    Nice-to-have:
    - Learning-to-rank models experience
    - HR-tech or recruiting platform experience
    - Distributed systems experience
    
    Disqualifiers:
    - Pure research background without production deployment
    - Only recent LangChain/OpenAI projects (< 12 months)
    - No production code in last 18 months
    """
    
    jd = JobDescription(
        raw_text=jd_text,
        paragraph_count=15
    )
    
    # Load sample candidates
    with open('inputs/datasets/sample_candidates.json', 'r') as f:
        sample_data = json.load(f)
    
    candidates = [Candidate(**c) for c in sample_data[:5]]  # First 5 candidates
    
    # Run evaluator
    evaluator = RelevanceEvaluator()
    result = evaluator.evaluate_all_candidates(jd, candidates)
    
    # Print results
    print("\n" + "="*80)
    print("RELEVANCE EVALUATION RESULTS")
    print("="*80)
    print(f"\nJD Summary:\n{result.jd_summary}\n")
    
    print(f"Evaluated {len(result.evaluations)} candidates\n")
    
    for eval_res in result.evaluations:
        print(f"\nCandidate: {eval_res.candidate_id}")
        print(f"  Vector Similarity:     {eval_res.vector_similarity.cosine_similarity:.3f}")
        print(f"  Rule-Based Score:      {eval_res.rule_based_score}/100")
        print(f"  Combined Score:        {eval_res.combined_score:.3f}")
    
    print("\n" + "="*80)
    print("RANKED ORDER (by combined score):")
    print("="*80)
    for i, eval_res in enumerate(result.evaluations, 1):
        print(f"{i}. {eval_res.candidate_id}: {eval_res.combined_score:.3f}")


if __name__ == "__main__":
    test_relevance_evaluation()
