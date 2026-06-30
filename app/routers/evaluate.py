import json
from typing import Optional

from fastapi import APIRouter, HTTPException, File, Form, UploadFile
from pydantic import ValidationError

from app.config import get_settings
from app.models.candidate import Candidate
from app.models.preprocessed import PreprocessedInput
from app.services.docx_parser import extract_jd_text
from app.services.dynamic_evaluation_service import get_dynamic_evaluation_service

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
    downstream relevance and behavioral trajectory analysis.
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


from app.models.evaluation import EvaluationResult
from app.services.relevance_evaluator import RelevanceEvaluator

evaluator = RelevanceEvaluator()


@router.post("/evaluate-relevance", response_model=EvaluationResult)
async def evaluate_relevance(
    jd_file: UploadFile = File(..., description="The Job Description as a .docx file"),
    candidates_json: str = Form(..., description="JSON array of candidates matching the Candidate schema")
):
    """
    Comprehensive relevance evaluation combining vector similarity and rule-based analysis.
    
    For each candidate:
    1. Generates embeddings for JD and candidate profile
    2. Computes cosine similarity score (0.0-1.0)
    3. Analyzes skills/experience against JD requirements (0-100)
    4. Combines into a final relevance score
    
    Returns ranked evaluations with justifications.
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
    if len(jd_bytes) > max_bytes:
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
    
    # --- Run evaluation ---
    evaluation_result = evaluator.evaluate_all_candidates(job_description, candidates)
    
    return evaluation_result




# NEW DYNAMIC ENDPOINTS - Support various JD and candidate formats


@router.post("/evaluate-dynamic", response_model=EvaluationResult)
async def evaluate_dynamic(
    jd_text: Optional[str] = Form(None, description="Job description as plain text"),
    jd_file: Optional[UploadFile] = File(None, description="Job description as .docx file"),
    candidates_json: Optional[str] = Form(None, description="Candidates as JSON array"),
    candidates_jsonl: Optional[str] = Form(None, description="Candidates as JSONL (one per line)"),
    format_hint: Optional[str] = Form(None, description="Format hint: 'json_array' or 'jsonl'")
):
    """
    Dynamic evaluation endpoint supporting multiple input formats.
    
    Supports:
    - JD: Plain text OR .docx file
    - Candidates: JSON array OR JSONL format
    
    Returns ranked candidates with FastEmbed embeddings + rule-based scores.
    """
    settings = get_settings()
    service = get_dynamic_evaluation_service()

    # Process JD
    if not jd_text and not jd_file:
        raise HTTPException(status_code=400, detail="Provide either jd_text or jd_file")

    if jd_file:
        if not jd_file.filename.lower().endswith(".docx"):
            raise HTTPException(status_code=400, detail="JD file must be .docx")
        
        jd_bytes = await jd_file.read()
        max_bytes = settings.max_upload_size_mb * 1024 * 1024
        if len(jd_bytes) > max_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"JD file exceeds {settings.max_upload_size_mb}MB limit"
            )
        
        try:
            jd_input = jd_bytes
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"JD processing failed: {str(e)}")
    else:
        jd_input = jd_text

    # Process candidates
    if not candidates_json and not candidates_jsonl:
        raise HTTPException(status_code=400, detail="Provide either candidates_json or candidates_jsonl")

    candidates_input = candidates_json or candidates_jsonl
    format_hint = format_hint or ('json_array' if candidates_json else 'jsonl')

    # Evaluate
    try:
        result = service.evaluate(jd_input, candidates_input, format_hint)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.post("/validate-candidates")
async def validate_candidates(
    candidates_json: str = Form(..., description="Candidates as JSON array or JSONL")
):
    """
    Validate candidates against schema without evaluation.
    
    Returns validation results for each candidate.
    """
    service = get_dynamic_evaluation_service()

    try:
        candidates = service.parse_candidates(candidates_json)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "valid": True,
        "candidate_count": len(candidates),
        "candidates": [
            {"candidate_id": c.candidate_id, "valid": True}
            for c in candidates
        ]
    }


@router.post("/jd-analysis")
async def analyze_jd(
    jd_text: Optional[str] = Form(None, description="JD as plain text"),
    jd_file: Optional[UploadFile] = File(None, description="JD as .docx file")
):
    """
    Analyze JD and extract keywords without evaluating candidates.
    
    Returns JD summary with extracted keywords.
    """
    settings = get_settings()
    service = get_dynamic_evaluation_service()

    if not jd_text and not jd_file:
        raise HTTPException(status_code=400, detail="Provide either jd_text or jd_file")

    if jd_file:
        if not jd_file.filename.lower().endswith(".docx"):
            raise HTTPException(status_code=400, detail="JD file must be .docx")
        
        jd_bytes = await jd_file.read()
        max_bytes = settings.max_upload_size_mb * 1024 * 1024
        if len(jd_bytes) > max_bytes:
            raise HTTPException(status_code=400, detail=f"JD exceeds {settings.max_upload_size_mb}MB")
        
        jd_input = jd_bytes
    else:
        jd_input = jd_text

    try:
        jd = service.process_jd(jd_input)
        summary = service.get_jd_summary(jd)
        return summary
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
