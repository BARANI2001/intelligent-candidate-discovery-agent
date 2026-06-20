"""
App entry point. Run with:

    uvicorn app.main:app --reload

`--reload` restarts the server automatically when you edit code --
useful in development, remove it in production.
"""

from fastapi import FastAPI
from app.routers import evaluate

app = FastAPI(
    title="Intelligent Candidate Discovery Agent",
    description="Evaluates and ranks candidates against a Job Description using LLMs and vector similarity.",
    version="0.1.0",
)

app.include_router(evaluate.router)

@app.get("/health")
async def health():
    return {"status": "ok"}