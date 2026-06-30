import json
import asyncio
import uuid
import logging
from threading import Lock
from typing import Optional, List, Dict, Any
import numpy as np

from fastapi import APIRouter, HTTPException, File, Form, UploadFile
from pydantic import ValidationError, BaseModel, Field

logger = logging.getLogger(__name__)

from app.config import get_settings
from app.models.candidate import Candidate
from app.models.jd import JobDescription
from app.models.preprocessed import PreprocessedInput
from app.models.ranking import RankingResponse, JobStatusResponse
from app.models.evaluation import EvaluationResult
from app.services.docx_parser import extract_jd_text
from app.services.dynamic_evaluation_service import get_dynamic_evaluation_service
from app.services.behavioral_evaluator import BehavioralEvaluator
from app.services.ranking_service import ConsolidationService
from app.services.relevance_evaluator import RelevanceEvaluator

router = APIRouter()
evaluator = RelevanceEvaluator()
behavioral_evaluator = BehavioralEvaluator()
consolidation_service = ConsolidationService()

# In-memory store for async background ranking jobs
jobs_store: Dict[str, Dict[str, Any]] = {}


class ParentJob:
    def __init__(self, jd: JobDescription, requirements: Dict[str, Any], jd_embedding: np.ndarray, jd_summary: str):
        self.jd = jd
        self.requirements = requirements
        self.jd_embedding = jd_embedding
        self.jd_summary = jd_summary
        self.candidates: List[Candidate] = []
        self.relevance_evals = []
        self.behavioral_evals = []
        self.lock = Lock()


class JobInitResponse(BaseModel):
    parent_job_id: str
    status: str
    jd_summary: str


class ChunkUploadResponse(BaseModel):
    status: str
    candidates_processed: int
    total_candidates_collected: int


parent_jobs_store: Dict[str, ParentJob] = {}

async def helper_extract_jd(jd_file: UploadFile) -> JobDescription:
    """
    Validate and extract text from an uploaded .docx Job Description file
    using DynamicEvaluationService / docx_parser.
    """
    settings = get_settings()
    if not jd_file.filename.lower().endswith(".docx"):
        raise HTTPException(
            status_code=400,
            detail="The JD file must be a .docx file."
        )
    
    jd_bytes = await jd_file.read()
    # max_bytes = settings.max_upload_size_mb * 1024 * 1024
    # if len(jd_bytes) > max_bytes:
    #     raise HTTPException(
    #         status_code=400,
    #         detail=f"JD file exceeds the {settings.max_upload_size_mb}MB limit."
    #     )
    
    service = get_dynamic_evaluation_service()
    try:
        return service.process_jd(jd_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"JD processing failed: {str(e)}")


def helper_parse_candidates(candidates_json: str) -> List[Candidate]:
    """
    Parse and validate candidates input string (JSON array or JSONL)
    using DynamicEvaluationService schema validation.
    """
    service = get_dynamic_evaluation_service()
    try:
        candidates = service.parse_candidates(candidates_json)
        if not candidates:
            raise HTTPException(
                status_code=400,
                detail="The candidates array is empty. Please provide at least one candidate."
            )
        return candidates
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/evaluate-candidates", response_model=PreprocessedInput)
async def evaluate_candidates(
    jd_file: UploadFile = File(..., description="The Job Description as a .docx file"),
    candidates_json: str = Form(..., description="JSON array of candidates matching the Candidate schema")
):
    """
    Endpoint to process a Job Description (JD) and a list of candidates.
    """
    job_description = await helper_extract_jd(jd_file)
    candidates = helper_parse_candidates(candidates_json)
    
    return PreprocessedInput(
        job_description=job_description,
        candidates=candidates
    )


@router.post("/evaluate-relevance", response_model=EvaluationResult)
async def evaluate_relevance(
    jd_file: UploadFile = File(..., description="The Job Description as a .docx file"),
    candidates_json: str = Form(..., description="JSON array of candidates matching the Candidate schema")
):
    """
    Comprehensive relevance evaluation combining vector similarity and rule-based analysis.
    """
    job_description = await helper_extract_jd(jd_file)
    candidates = helper_parse_candidates(candidates_json)
    
    evaluation_result = evaluator.evaluate_all_candidates(job_description, candidates)
    return evaluation_result


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
    """
    service = get_dynamic_evaluation_service()

    if jd_file:
        job_description = await helper_extract_jd(jd_file)
        jd_input = job_description.raw_text
    elif jd_text:
        jd_input = jd_text
    else:
        raise HTTPException(status_code=400, detail="Provide either jd_text or jd_file")

    candidates_input = candidates_json or candidates_jsonl
    if not candidates_input:
        raise HTTPException(status_code=400, detail="Provide either candidates_json or candidates_jsonl")

    try:
        return service.evaluate(jd_input, candidates_input, format_hint)
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
    """
    candidates = helper_parse_candidates(candidates_json)

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
    """
    service = get_dynamic_evaluation_service()

    if jd_file:
        jd = await helper_extract_jd(jd_file)
    elif jd_text:
        jd = service.process_jd(jd_text)
    else:
        raise HTTPException(status_code=400, detail="Provide either jd_text or jd_file")

    try:
        summary = service.get_jd_summary(jd)
        return summary
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


async def run_ranking_background_task(job_id: str, job_description: JobDescription, candidates: List[Candidate]):
    """
    Background worker: runs relevance + behavioral evaluation concurrently.
    - Relevance: one embed_texts() call for all candidates + parallel rule scoring (ThreadPoolExecutor)
    - Behavioral: parallel heuristics (ThreadPoolExecutor)
    Both run in separate threads simultaneously via asyncio.gather.
    """
    total = len(candidates)
    logger.info(f"[Job {job_id}] Started ranking {total} candidates.")

    try:
        logger.info(f"[Job {job_id}] Running relevance + behavioral evaluation in parallel...")
        relevance_result, behavioral_result = await asyncio.gather(
            asyncio.to_thread(evaluator.evaluate_all_candidates, job_description, candidates),
            asyncio.to_thread(behavioral_evaluator.evaluate_all_candidates, candidates)
        )

        logger.info(f"[Job {job_id}] Consolidating final ranking...")
        ranking_response = consolidation_service.consolidate_and_rank(
            candidates=candidates,
            relevance_evals=relevance_result.evaluations,
            behavioral_evals=behavioral_result,
            jd_summary=relevance_result.jd_summary
        )

        jobs_store[job_id] = {
            "status": "completed",
            "results": ranking_response,
            "error": None
        }
        logger.info(f"[Job {job_id}] Successfully ranked {total} candidates.")

    except Exception as e:
        logger.error(f"[Job {job_id}] Failed: {str(e)}", exc_info=True)
        jobs_store[job_id] = {
            "status": "failed",
            "results": None,
            "error": str(e)
        }



@router.post("/rank-candidates", response_model=JobStatusResponse)
async def rank_candidates(
    jd_file: UploadFile = File(..., description="The Job Description as a .docx file"),
    candidates_json: str = Form(..., description="JSON array of candidates matching the Candidate schema")
):
    """
    End-to-End Candidate Ranking Pipeline (Async Background Execution).
    Returns a job tracking object immediately. Poll GET /ranking-status/{job_id} every 5 seconds.
    """
    job_description = await helper_extract_jd(jd_file)
    candidates = helper_parse_candidates(candidates_json)

    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "status": "processing",
        "results": None,
        "error": None
    }

    asyncio.create_task(run_ranking_background_task(job_id, job_description, candidates))

    return JobStatusResponse(
        job_id=job_id,
        status="processing"
    )


@router.get("/ranking-status/{job_id}", response_model=JobStatusResponse)
async def get_ranking_status(job_id: str):
    """
    Poll the status and results of an asynchronous candidate ranking job.
    """
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job ID not found")

    job = jobs_store[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        error=job.get("error"),
        results=job.get("results")
    )


@router.post("/jobs/init", response_model=JobInitResponse)
async def init_parent_job(
    jd_file: UploadFile = File(..., description="The Job Description as a .docx file")
):
    """
    Initialize a parent job with a Job Description. Pre-computes and caches requirements.
    """
    job_description = await helper_extract_jd(jd_file)
    
    # Pre-extract requirements and embedding
    requirements = evaluator.extract_jd_requirements(job_description)
    jd_embedding = evaluator.generate_jd_embedding(job_description)
    
    title_str = " ".join(requirements["title_keywords"]).title()
    jd_summary = f"""
Job Role: {title_str}
Experience: {requirements['experience']['min']}-{requirements['experience']['max']} years
Must-have: {", ".join(requirements['must_have'][:5])}
Nice-to-have: {", ".join(requirements['nice_to_have'][:5])}
    """.strip()
    
    parent_job_id = str(uuid.uuid4())
    parent_jobs_store[parent_job_id] = ParentJob(
        jd=job_description,
        requirements=requirements,
        jd_embedding=jd_embedding,
        jd_summary=jd_summary
    )
    
    logger.info(f"Initialized parent job {parent_job_id}")
    return JobInitResponse(
        parent_job_id=parent_job_id,
        status="initialized",
        jd_summary=jd_summary
    )


@router.post("/jobs/{parent_job_id}/chunks", response_model=ChunkUploadResponse)
async def upload_candidate_chunk(
    parent_job_id: str,
    candidates_json: str = Form(..., description="JSON array or JSONL of a chunk of candidates")
):
    """
    Upload and evaluate a chunk of candidates in parallel under a parent job.
    """
    if parent_job_id not in parent_jobs_store:
        raise HTTPException(status_code=404, detail="Parent Job ID not found")
        
    parent_job = parent_jobs_store[parent_job_id]
    chunk = helper_parse_candidates(candidates_json)
    
    logger.info(f"Processing chunk of {len(chunk)} candidates for parent job {parent_job_id}...")
    
    # Run evaluation of the chunk
    relevance_result, behavioral_result = await asyncio.gather(
        asyncio.to_thread(evaluator.evaluate_all_candidates, parent_job.jd, chunk),
        asyncio.to_thread(behavioral_evaluator.evaluate_all_candidates, chunk)
    )
    
    # Safely append results
    with parent_job.lock:
        parent_job.candidates.extend(chunk)
        parent_job.relevance_evals.extend(relevance_result.evaluations)
        parent_job.behavioral_evals.extend(behavioral_result)
        total_collected = len(parent_job.candidates)
        
    logger.info(f"Chunk of {len(chunk)} candidates processed. Total collected so far: {total_collected}")
    return ChunkUploadResponse(
        status="processed",
        candidates_processed=len(chunk),
        total_candidates_collected=total_collected
    )


@router.post("/jobs/{parent_job_id}/consolidate", response_model=RankingResponse)
@router.get("/jobs/{parent_job_id}/consolidate", response_model=RankingResponse)
async def consolidate_parent_job(parent_job_id: str):
    """
    Deduplicate, consolidate, and rank all candidate chunks uploaded to the parent job.
    """
    if parent_job_id not in parent_jobs_store:
        raise HTTPException(status_code=404, detail="Parent Job ID not found")
        
    parent_job = parent_jobs_store[parent_job_id]
    
    with parent_job.lock:
        if not parent_job.candidates:
            raise HTTPException(status_code=400, detail="No candidates have been uploaded for this parent job")
            
        logger.info(f"Consolidating and ranking {len(parent_job.candidates)} total candidates for parent job {parent_job_id}...")
        
        # Deduplicate candidates and evaluations by candidate_id to avoid mistakes
        seen = set()
        deduped_candidates = []
        
        # Keep latest evaluation
        relevance_by_id = {r.candidate_id: r for r in parent_job.relevance_evals}
        behavioral_by_id = {b.candidate_id: b for b in parent_job.behavioral_evals}
        
        for c in parent_job.candidates:
            if c.candidate_id not in seen:
                seen.add(c.candidate_id)
                deduped_candidates.append(c)
                
        deduped_relevance = [relevance_by_id[c.candidate_id] for c in deduped_candidates if c.candidate_id in relevance_by_id]
        deduped_behavioral = [behavioral_by_id[c.candidate_id] for c in deduped_candidates if c.candidate_id in behavioral_by_id]
        
        ranking_response = consolidation_service.consolidate_and_rank(
            candidates=deduped_candidates,
            relevance_evals=deduped_relevance,
            behavioral_evals=deduped_behavioral,
            jd_summary=parent_job.jd_summary
        )
        
    logger.info(f"Successfully consolidated {len(deduped_candidates)} candidates for parent job {parent_job_id}.")
    return ranking_response

