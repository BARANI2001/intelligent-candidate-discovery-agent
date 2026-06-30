from pydantic import BaseModel, Field
from typing import List


class VectorSimilarityResult(BaseModel):
    """Result of vector embedding and cosine similarity computation."""
    cosine_similarity: float = Field(
        ge=0.0, 
        le=1.0, 
        description="Cosine similarity score between JD and candidate profile embeddings."
    )


class RelevanceEvaluation(BaseModel):
    """Relevance evaluation for a single candidate against a JD."""
    candidate_id: str
    vector_similarity: VectorSimilarityResult
    rule_based_score: int = Field(
        ge=0,
        le=100,
        description="Rule-based skill and experience match score (0-100)."
    )
    combined_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Weighted combination of vector similarity and normalized rule-based score."
    )


class BehavioralEvaluation(BaseModel):
    """Behavioral and career trajectory evaluation for a single candidate."""
    candidate_id: str
    tenure_score: float = Field(ge=0, le=100, description="Score based on average tenure and job hopping.")
    progression_score: float = Field(ge=0, le=100, description="Score based on seniority promotions or demotions.")
    composite_behavioral_score: float = Field(ge=0, le=100, description="Combined behavioral score (0-100).")
    total_duration_months: int
    avg_tenure_months: float
    job_hopping_penalty_applied: bool
    demotions_detected: int
    summary: str


class EvaluationResult(BaseModel):
    """Complete evaluation result for all candidates."""
    evaluations: List[RelevanceEvaluation] = Field(
        description="Relevance evaluation for each candidate."
    )
    jd_summary: str = Field(
        description="Summary of key JD requirements."
    )