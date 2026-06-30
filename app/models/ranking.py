"""
Ranking and Consolidation Response Models.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class RankedCandidate(BaseModel):
    candidate_id: str
    rank: int = Field(ge=1, description="Integer rank starting from 1 for the highest scoring candidate.")
    final_score: float = Field(description="Modified final score combining relevance, behavior, and Redrob signals.")
    base_score: float = Field(description="60% relevance score + 40% behavioral score before signal modifiers.")
    relevance_score: float = Field(description="Rule-based relevance match score (0-100).")
    behavioral_score: float = Field(description="Algorithmic behavioral and career trajectory score (0-100).")
    applied_multipliers: Dict[str, Any] = Field(description="Dictionary detailing applied signal multipliers and boosts.")
    justification: str = Field(description="Deterministically generated explanation string detailing score breakdowns.")
    relevance_justification: Optional[str] = None
    behavioral_summary: Optional[str] = None


class RankingResponse(BaseModel):
    job_description_summary: str = Field(description="Summary of key JD requirements.")
    total_candidates: int = Field(description="Total number of evaluated candidates.")
    results: List[RankedCandidate] = Field(description="List of candidates sorted descending by final_score.")


class JobStatusResponse(BaseModel):
    job_id: str = Field(description="Unique identifier for the asynchronous ranking job.")
    status: str = Field(description="Current job status: 'processing', 'completed', or 'failed'.")
    error: Optional[str] = Field(None, description="Error message if the job failed.")
    results: Optional[RankingResponse] = Field(None, description="Ranking response once the job completes.")