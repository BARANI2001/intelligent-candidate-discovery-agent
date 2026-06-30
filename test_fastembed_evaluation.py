#!/usr/bin/env python3
"""
Test script using FastEmbed embeddings with real sample data.

Workflow:
1. Extract job description keywords from JD text
2. Generate FastEmbed embeddings for JD keywords
3. Generate FastEmbed embeddings for candidate skills
4. Compute cosine similarity between them
5. Combine with Rule-Based Relevance Score for final ranking
"""

import json
import traceback
from app.models.candidate import Candidate
from app.models.jd import JobDescription
from app.services.relevance_evaluator import RelevanceEvaluator
from app.services.embedding_service import get_embedding_service
from app.services.docx_parser import extract_jd_text


def test_fastembed_workflow():
    """Test complete FastEmbed embedding workflow."""
    
    print("\n" + "="*80)
    print("FASTEMBED EMBEDDING-BASED RELEVANCE EVALUATION")
    print("="*80)
    
    # Load sample JD and candidates
    print("\n[1] Loading Job Description...")
    
    # Create sample JD for testing
    jd_text = """
    Senior AI Engineer — Founding Team
    Location: Pune/Noida, India
    Experience Required: 5–9 years
    
    Key Requirements:
    - Production experience with embeddings-based retrieval systems
    - Strong experience with vector databases (Pinecone, Weaviate, Qdrant, Milvus)
    - Strong Python skills and experience with ML frameworks
    - Experience designing evaluation frameworks for ranking systems (NDCG, MRR, MAP)
    - Understanding of hybrid retrieval vs dense retrieval
    
    Nice-to-have:
    - Learning-to-rank models experience
    - HR-tech or recruiting platform experience
    - Distributed systems experience (Kafka, Spark)
    - Open-source contributions on GitHub
    
    Disqualifiers:
    - Pure research background without production deployment
    - Only recent LangChain/OpenAI projects
    - No production code in last 18 months
    """
    
    jd = JobDescription(raw_text=jd_text, paragraph_count=15)
    print(f"[PASS] JD loaded ({len(jd_text)} chars)")
    
    # Load sample candidates
    print("\n[2] Loading Sample Candidates...")
    
    with open('inputs/datasets/sample_candidates.json', 'r') as f:
        sample_data = json.load(f)
    
    candidates = [Candidate(**c) for c in sample_data[:5]]
    print(f"[PASS] Loaded {len(candidates)} candidates from sample data")
    
    # Initialize evaluator with FastEmbed
    print("\n[3] Initializing FastEmbed Embedding Service...")
    
    evaluator = RelevanceEvaluator()
    embedding_service = get_embedding_service()
    print(f"[PASS] FastEmbed initialized")
    
    # Extract JD keywords
    print("\n[4] Extracting JD Keywords...")
    
    jd_keywords = evaluator.extract_jd_requirements(jd)
    print(f"[PASS] Extracted {len(jd_keywords)} keywords from JD:")
    for i, keyword in enumerate(jd_keywords[:10], 1):
        print(f"   {i:2d}. {keyword}")
    if len(jd_keywords) > 10:
        print(f"   ... and {len(jd_keywords) - 10} more")
    
    # Generate JD keyword embeddings
    print("\n[5] Generating JD Keyword Embeddings...")
    
    jd_embedding = embedding_service.embed_jd_keywords(jd_keywords)
    print(f"[PASS] JD keywords embedded (shape: {jd_embedding.shape})")
    print(f"   Embedding sample: {jd_embedding[:5]}")
    
    # Evaluate each candidate
    print("\n[6] Evaluating Candidates with FastEmbed + Rule-Based Scoring...\n")
    
    results = []
    
    for idx, candidate in enumerate(candidates, 1):
        print(f"Candidate {idx}: {candidate.candidate_id}")
        print(f"  Profile: {candidate.profile.current_title} at {candidate.profile.current_company}")
        print(f"  Experience: {candidate.profile.years_of_experience} years")
        
        # Get candidate skills
        candidate_skills = [s.name for s in candidate.skills]
        print(f"  Skills ({len(candidate_skills)}): {', '.join(candidate_skills[:5])}...")
        
        # Generate candidate skills embedding
        candidate_embedding = embedding_service.embed_candidate_skills(candidate_skills)
        print(f"  Candidate embedding shape: {candidate_embedding.shape}")
        
        # Compute embedding-based cosine similarity
        embedding_similarity = embedding_service.cosine_similarity(jd_embedding, candidate_embedding)
        print(f"  [SIM] Embedding Similarity (FastEmbed): {embedding_similarity:.3f}")
        
        # Compute vector similarity using full profile
        vector_sim_result = evaluator.compute_vector_similarity(jd, candidate)
        print(f"  [SIM] Profile Embedding Similarity: {vector_sim_result.cosine_similarity:.3f}")
        
        # Get evaluation result (includes rule-based score)
        eval_result = evaluator.evaluate_candidate(jd, candidate)
        print(f"  [SCORE] Rule-Based Score: {eval_result.rule_based_score}/100")
        print(f"  [SCORE] Combined Score: {eval_result.combined_score:.3f}")
        
# Get rule-based factors for display
        rule_factors = evaluator._compute_rule_factors(jd, candidate)
        
        results.append({
            'candidate_id': candidate.candidate_id,
            'embedding_similarity': embedding_similarity,
            'profile_similarity': vector_sim_result.cosine_similarity,
            'rule_based_score': eval_result.rule_based_score,
            'combined_score': eval_result.combined_score,
            'justification': rule_factors
        })
        
        print()
    
    # Rank by combined score
    print("="*80)
    print("RANKING BY COMBINED SCORE (Embedding + Rule-Based)")
    print("="*80)
    
    results_sorted = sorted(results, key=lambda x: x['combined_score'], reverse=True)
    
    for rank, result in enumerate(results_sorted, 1):
        print(f"\n{rank}. {result['candidate_id']}")
        print(f"   Embedding Similarity:  {result['embedding_similarity']:.3f}")
        print(f"   Profile Similarity:    {result['profile_similarity']:.3f}")
        print(f"   Rule-Based Score:      {result['rule_based_score']}/100")
        print(f"   Combined Score:        {result['combined_score']:.3f}")
        print(f"   Top Factors:           {result['justification']}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    embedding_sims = [r['embedding_similarity'] for r in results]
    profile_sims = [r['profile_similarity'] for r in results]
    rule_scores = [r['rule_based_score'] for r in results]
    combined_scores = [r['combined_score'] for r in results]
    
    print(f"\nEmbedding Similarity (FastEmbed):")
    print(f"  Mean: {sum(embedding_sims)/len(embedding_sims):.3f}")
    print(f"  Range: {min(embedding_sims):.3f} - {max(embedding_sims):.3f}")
    
    print(f"\nProfile Similarity:")
    print(f"  Mean: {sum(profile_sims)/len(profile_sims):.3f}")
    print(f"  Range: {min(profile_sims):.3f} - {max(profile_sims):.3f}")
    
    print(f"\nRule-Based Scores:")
    print(f"  Mean: {sum(rule_scores)/len(rule_scores):.1f}/100")
    print(f"  Range: {min(rule_scores)}/100 - {max(rule_scores)}/100")
    
    print(f"\nCombined Scores:")
    print(f"  Mean: {sum(combined_scores)/len(combined_scores):.3f}")
    print(f"  Range: {min(combined_scores):.3f} - {max(combined_scores):.3f}")
    
    print("\n" + "="*80)
    print("[PASS] FastEmbed Evaluation Complete!")
    print("="*80 + "\n")
    
    return results_sorted


def test_embedding_quality():
    """Test embedding quality and semantic understanding."""
    
    print("\n" + "="*80)
    print("EMBEDDING QUALITY TEST")
    print("="*80)
    
    embedding_service = get_embedding_service()
    
    # Test 1: Semantic similarity of synonyms
    print("\n1. Semantic Similarity of Related Skills:\n")
    
    skill_pairs = [
        ("Machine Learning", "Deep Learning"),
        ("Python", "PyTorch"),
        ("Vector Database", "Semantic Search"),
        ("Embeddings", "Retrieval"),
        ("Ranking", "Information Retrieval"),
    ]
    
    for skill1, skill2 in skill_pairs:
        emb1 = embedding_service.embed_text(skill1)
        emb2 = embedding_service.embed_text(skill2)
        similarity = embedding_service.cosine_similarity(emb1, emb2)
        print(f"  '{skill1}' ↔ '{skill2}': {similarity:.3f}")
    
    # Test 2: Batch embedding performance
    print("\n2. Batch Embedding Performance:\n")
    
    skills = [
        "Python", "Machine Learning", "Vector Databases",
        "Embeddings", "Ranking Systems", "NLP"
    ]
    
    embeddings = embedding_service.embed_texts(skills)
    print(f"  Embedded {len(skills)} skills in batch mode")
    print(f"  Embedding dimensions: {embeddings[0].shape if embeddings else 'N/A'}")
    
    # Mean pooling
    mean_embedding = embedding_service.embed_candidate_skills(skills)
    print(f"  Mean-pooled embedding shape: {mean_embedding.shape}")


if __name__ == "__main__":
    try:
        results = test_fastembed_workflow()
        test_embedding_quality()
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        traceback.print_exc()
