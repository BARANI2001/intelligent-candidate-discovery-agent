from pydantic import BaseModel
from typing import List

from app.models.candidate import Candidate
from app.models.jd import JobDescription

class PreprocessedInput(BaseModel):
    job_description: JobDescription
    candidates: List[Candidate]