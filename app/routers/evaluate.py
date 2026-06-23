import json

from fastapi import APIRouter, HTTPException, File, Form, UploadFile
from pydantic import ValidationError

from app.config import get_settings
from app.models.candidate import Candidate
from app.models.preprocessed import PreprocessedInput
from app.services.docx_parser import extract_jd_text

router = APIRouter()

@router.post("/evaluate-candidates", response_model=PreprocessedInput)
async def evaluate_candidates(
    jd_file: UploadFile = File(..., description="The Job Description as a .docx file"),
    candidates_json: str = Form(..., description="JSON array of candidates matching the Candidate schema")
):
    """
    Endpoint to process a Job Description (JD) and a list of candidates.
    
    This endpoint validates the JD file format, extracts its text, and parses
    the provided JSON string into a list of Candidate models. It returns
    a PreprocessedInput object containing the combined data ready for
    downstream LLM relevance and behavioral trajectory analysis.
    """
    settings = get_settings()

    # --- Validate the JD file ---
    if not jd_file.filename.lower().endswith(".docx"):
        raise HTTPException(
            status_code=400,
            detail="The JD file must be a .docx file."
        )
    
    jd_bytes = await jd_file.read()

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if (len(jd_bytes) > max_bytes):
        raise HTTPException(
            status_code=400,
            detail=f"JD file exceeds the {settings.max_upload_size_mb}MB limit."
        )
    
    # --- Extract JD text ---
    job_description = extract_jd_text(jd_bytes)

    # --- Parse and Validate candidates JSON ---
    try:
        raw_candidates = json.loads(candidates_json)
    except json.JSONDecodeError as exception:
        raise HTTPException(
            status_code=400,
            detail=f"candidates_json is not valid JSON: {exception}"
        )
    
    if not isinstance(raw_candidates, list):
        raise HTTPException(
            status_code=400,
            detail="candidates_json must be a JSON array of candidate objects."
        )
    
    try:
        candidates = [Candidate(**c) for c in raw_candidates]
    except ValidationError as exception:
        raise HTTPException(
            status_code=422,
            detail=exception.errors()
        )
    
    if not candidates:
        raise HTTPException(
            status_code=400,
            detail="The candidates_json array is empty. Please provide at least one candidate."
        )
    
    return PreprocessedInput(
        job_description=job_description,
        candidates=candidates
    )