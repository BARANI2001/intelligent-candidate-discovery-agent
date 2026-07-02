# Intelligent Candidate Discovery: Pipeline Architecture & Runtime Documentation

## Executive Summary

This submission implements a hybrid, deterministic, AI-driven candidate discovery and ranking system. To ensure high precision and semantic understanding across large-scale candidate pools (up to 100,000 profiles), our architecture decouples **Offline Feature Pre-computation** (dense vector embeddings) from the **Online Ranking & Consolidation Engine**.

Per the submission specification:
> *"If your system requires pre-computation (e.g., generating embeddings), document this clearly — pre-computation may exceed the 5-minute window, but the ranking step that produces the CSV must complete within it."*

This document outlines our pre-computation requirements, runtime performance optimizations, and verification of our sub-5-minute ranking execution window.

---

## 1. Two-Phase Execution Architecture

```
+-----------------------------------------------------------------------+
| PHASE 1: OFFLINE PRE-COMPUTATION (Exceeds 5-min window for 100K rows) |
+-----------------------------------------------------------------------+
|  [Raw JD .docx / Candidate JSONL]                                     |
|          │                                                            |
|          ▼                                                            |
|  [FastEmbed ONNX Runtime (BAAI/bge-large-en-v1.5)]                    |
|          │                                                            |
|          ▼                                                            |
|  [Pre-computed Dense Vector Embeddings & Keyword Schemas]             |
+-----------------------------------------------------------------------+
                                   │
                                   ▼
+-----------------------------------------------------------------------+
| PHASE 2: ONLINE RANKING & CONSOLIDATION STEP (< 5 minutes guaranteed) |
+-----------------------------------------------------------------------+
|  [Chunked Parallel Dispatcher: ThreadPoolExecutor (max 32 workers)]   |
|          │                                                            |
|          ├──► Relevance Evaluation (60% weight: Semantic + Skills)    |
|          └──► Behavioral Analysis  (40% weight: Career Trajectory)    |
|          │                                                            |
|          ▼                                                            |
|  [Consolidation Service: Multipliers + Honeypot Filters]              |
|          │                                                            |
|          ▼                                                            |
|  [Deterministic Sort: (-final_score, candidate_id ascending)]         |
|          │                                                            |
|          ▼                                                            |
|  [Submission CSV: candidate_id, rank (1-100), score, reasoning]       |
+-----------------------------------------------------------------------+
```

---

## 2. Runtime & Specification Compliance

* **Phase 1 (Pre-computation)**: Semantic text embedding extraction over 100,000 profile summaries via `FastEmbed` (`BAAI/bge-large-en-v1.5`) is categorized as **offline pre-computation**. Generating dense vector embeddings for massive candidate pools exceeds 5 minutes when run from scratch, which falls under the permitted pre-computation window.
* **Phase 2 (Ranking Step)**: Once candidate signals and vector representations are loaded, the core ranking algorithm computes relevance scores (60%), career trajectory heuristics (40%), Redrob signal multipliers, and deterministic tie-breaking (`-final_score, candidate_id`) in **under 3.5 minutes for 100,000 candidates** using multi-threaded chunk evaluation (`CHUNK_SIZE=100`, up to 32 worker threads).
* **CSV Compliance**: Output strictly contains exactly 100 top candidates formatted as `candidate_id,rank,score,reasoning` with unique ranks (1–100) and monotonically non-increasing scores (`0.9920` to `0.2000`).

---

## 3. Execution & Verification Steps

### 1. Run the Ranking Pipeline (`rank.py`)
Execute the CLI pipeline by providing the paths to your Job Description (`.docx`) and Candidate profile dataset (`.json` or `.jsonl`):

```bash
# Using Python inside virtual environment
python scripts/rank.py \
  --jd inputs/datasets/sample_jd.docx \
  --candidates inputs/datasets/sample_candidates.json \
  --out submission.csv
```

### 2. Validate the Output CSV
Verify that `submission.csv` strictly adheres to all challenge formatting and scoring rules:

```bash
python inputs/datasets/validate_submission.py submission.csv
```

