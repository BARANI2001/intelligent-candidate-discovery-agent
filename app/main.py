"""
App entry point. Run with:

    uvicorn app.main:app --reload

`--reload` restarts the server automatically when you edit code --
useful in development, remove it in production.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import starlette.formparsers
from app.routers import evaluate

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