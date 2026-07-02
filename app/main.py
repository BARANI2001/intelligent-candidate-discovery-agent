"""
App entry point. Run with:

    uvicorn app.main:app --reload

`--reload` restarts the server automatically when you edit code --
useful in development, remove it in production.
"""

from contextlib import asynccontextmanager
import logging
import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import starlette.formparsers
from app.routers import evaluate
from app.services.embedding_service import get_embedding_service
from app.services.docx_parser import extract_jd_text

# Explicitly configure the parent 'app' logger to output INFO logs to stdout
app_logger = logging.getLogger("app")
app_logger.setLevel(logging.INFO)
if not app_logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    app_logger.addHandler(handler)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load embedding model before start serving requests
    logger.info("Pre-loading embedding model at startup...")
    try:
        service = get_embedding_service()
        service.load_model()
        logger.info("Embedding model loaded successfully.")
        
        # Load and embed default JD if it exists
        jd_path = "inputs/datasets/job_description.docx"
        if os.path.exists(jd_path):
            import time
            start_time = time.perf_counter()
            file_size = os.path.getsize(jd_path)
            logger.info(f"Starting pre-computation of JD from: {jd_path} (File size: {file_size} bytes)")
            
            with open(jd_path, "rb") as f:
                jd_bytes = f.read()
            
            jd = extract_jd_text(jd_bytes, filename=os.path.basename(jd_path))
            logger.info(f"Parsed default JD docx. Paragraph count: {jd.paragraph_count}, Text length: {len(jd.raw_text)} characters.")
            
            logger.info("Generating and caching default JD embedding...")
            service.cache_jd(jd.raw_text)
            
            from app.services.embedding_service import set_default_jd
            set_default_jd(jd)
            
            elapsed_time = time.perf_counter() - start_time
            logger.info(f"Pre-computation of default JD completed successfully in {elapsed_time:.4f} seconds.")
        else:
            logger.warning(f"Default JD path {jd_path} not found.")
            
    except Exception as e:
        logger.error(f"Failed to load embedding model or default JD at startup: {e}")
        raise e
    yield

# Monkeypatch Starlette form parsers to remove 1024KB limit
_orig_multipart_init = starlette.formparsers.MultiPartParser.__init__
_orig_form_init = starlette.formparsers.FormParser.__init__

def _new_multipart_init(self, *args, **kwargs):
    kwargs["max_part_size"] = 1024 * 1024 * 1024  # 1GB
    kwargs["max_files"] = 10000
    kwargs["max_fields"] = 10000
    _orig_multipart_init(self, *args, **kwargs)

def _new_form_init(self, *args, **kwargs):
    kwargs["max_part_size"] = 1024 * 1024 * 1024  # 1GB
    kwargs["max_fields"] = 10000
    _orig_form_init(self, *args, **kwargs)

starlette.formparsers.MultiPartParser.__init__ = _new_multipart_init
starlette.formparsers.FormParser.__init__ = _new_form_init

app = FastAPI(
    title="Intelligent Candidate Discovery Agent",
    description="Evaluates and ranks candidates against a Job Description using vector similarity.",
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:5174", "http://127.0.0.1:5174"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(evaluate.router)

@app.get("/health")
async def health():
    return {"status": "ok"}