from pydantic import BaseModel, Field
from typing import List

from app.models.candidate import Candidate
from app.models.jd import JobDescription

class PreprocessedInput(BaseModel):
    """The combined model returned by the /evaluate-candidates endpoint after parsing."""
    job_description: JobDescription = Field(description="The parsed and validated Job Description.")
    candidates: List[Candidate] = Field(description="The list of parsed and validated candidate profiles.")