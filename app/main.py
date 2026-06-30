"""
App entry point. Run with:

    uvicorn app.main:app --reload

`--reload` restarts the server automatically when you edit code --
useful in development, remove it in production.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import evaluate

app = FastAPI(
    title="Intelligent Candidate Discovery Agent",
    description="Evaluates and ranks candidates against a Job Description using vector similarity.",
    version="0.1.0",
)

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(evaluate.router)

@app.get("/health")
async def health():
    return {"status": "ok"}