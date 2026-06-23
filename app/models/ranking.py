"""
STUB — not implemented yet.

Will hold the response shape for (consolidation + ranking), e.g.:

    class RankedCandidate(BaseModel):
        candidate_id: str
        rank: int
        final_score: float
        base_score: float
        applied_multipliers: dict
        relevance_justification: str
        behavioral_summary: str

    class RankingResponse(BaseModel):
        results: list[RankedCandidate]
"""