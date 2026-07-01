# Intelligent Candidate Discovery Agent

Backend service for evaluating job descriptions against candidate profiles using FastAPI.

---

## Offline Evaluation (CLI — no server needed)

Run the full ranking pipeline directly from the terminal:

```bash
python scripts/rank.py \
  --jd ./job_description.docx \
  --candidates ./candidates.json \
  --out ./submission.csv
```

| Argument | Description |
|---|---|
| `--jd` | Path to the Job Description `.docx` file |
| `--candidates` | Path to candidates `.json` (array) or `.jsonl` file |
| `--out` | Output CSV path (default: `submission.csv`) |

**Output CSV columns** (submission-spec compliant):
```
candidate_id, rank, score, reasoning
```

> **Note:** Always run from the project root so the `app` package resolves correctly.

---

## API Server

See [`steps.md`](steps.md) for full setup instructions (prerequisites, environment, frontend).

```bash
# Install dependencies
uv sync

# Start backend (http://localhost:8000, Swagger at /docs)
make backend

# Start frontend (http://localhost:5173)
make ui
```

---

## Validate Submission CSV

```bash
python inputs/datasets/validate_submission.py your_submission.csv
```
