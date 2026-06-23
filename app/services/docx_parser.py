import io
from docx import Document
from fastapi import HTTPException

from app.models.jd import JobDescription

def extract_jd_text(file_bytes: bytes) -> JobDescription:
    """
    Takes the raw bytes of an uploaded .docx file and returns the 
    extracted text wrapped in a JobDescription model.
    
    Raises an HTTPException (which FastAPI turns into a proper error response) 
    if the file isn't a valid .docx -- e.g. someone uploaded 
    a .txt file renamed to .docx, or the upload got corrupted.
    """
    
    try:
        doc = Document(io.BytesIO(file_bytes))
    except Exception as exception:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read the uploaded file as a .docx document: {exception}"
        )
    
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

    if not paragraphs:
        raise HTTPException(
            status_code=400,
            detail="The uploaded .docx file appears to be empty."
        )
    
    raw_text = "\n\n".join(paragraphs)

    return JobDescription(
        raw_text=raw_text,
        paragraph_count=len(paragraphs)
    )