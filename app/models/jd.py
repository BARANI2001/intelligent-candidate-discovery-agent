from pydantic import BaseModel

class JobDescription(BaseModel):
    raw_text: str
    paragraph_count: int