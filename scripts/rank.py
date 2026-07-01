#!/usr/bin/env python3
"""
rank.py — Offline CLI runner for the Intelligent Candidate Discovery ranking pipeline.
Replicates the full end-to-end pipeline from app/routers/evaluate.py but runs
synchronously in the terminal without needing the FastAPI server.

Usage:
    python rank.py --jd <path/to/jd.docx> --candidates <path/to/candidates.json> --out <path/to/ranked_results.csv>

Arguments:
    --jd          Path to the Job Description .docx file (required)
    --candidates  Path to the candidates .json or .jsonl file (required)
    --out         Path for the output CSV file (default: ./submission.csv)

Output CSV columns (per submission spec):
    candidate_id, rank, score, reasoning

Chunked Parallel Processing
---------------------------
When the number of candidates exceeds CHUNK_THRESHOLD, the pipeline automatically
splits the candidate list into chunks of CHUNK_SIZE and evaluates them concurrently
across MAX_CONCURRENT_CHUNKS worker threads. All chunk results are then merged and
passed through a single consolidation + ranking pass.

Performance targets:
    - CHUNK_SIZE            : 100 candidates per chunk (threshold for parallel dispatch)
    - MAX_CONCURRENT_CHUNKS : min(32, cpu_count * 4) — balances CPU saturation with
                              GIL-release windows from FastEmbed (ONNX) and numpy.
    - Goal                  : 100 K candidates evaluated in < 5 minutes.
"""

import argparse
import csv
import os
import sys
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

# Ensure the project root (parent of this script's directory) is on sys.path
# so that `app.*` imports resolve correctly when running the script directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunked parallel processing configuration
# ---------------------------------------------------------------------------
# Number of candidates per evaluation chunk.  Keep at 100 so each chunk is
# small enough to be scheduled quickly while still amortising the FastEmbed
# batch-embedding overhead.
CHUNK_SIZE: int = 100

# Minimum total candidates before chunking kicks in.  Below this threshold the
# original single-batch path is used (avoids unnecessary thread overhead).
CHUNK_THRESHOLD: int = 100

# Maximum number of chunks that are evaluated concurrently.  FastEmbed's ONNX
# runtime releases the Python GIL during inference, so threads DO achieve real
# parallelism for the embedding step.  The rule-based + behavioral work is
# fast numpy / pure-Python.  Empirically, cpu_count * 4 saturates throughput
# without thrashing — capped at 32 to avoid excessive memory pressure when
# loading many embeddings simultaneously.
MAX_CONCURRENT_CHUNKS: int = min(32, (os.cpu_count() or 4) * 4)

# Maximum number of ranked candidates to write to the output CSV.
# Submission spec requires exactly 100 rows (1 header + top-100 candidates).
TOP_N_RESULTS: int = 100


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank candidates against a Job Description and produce a submission CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--jd",
        required=True,
        metavar="JD_FILE",
        help="Path to the Job Description (.docx) file.",
    )
    parser.add_argument(
        "--candidates",
        required=True,
        metavar="CANDIDATES_FILE",
        help="Path to the candidates file (.json array or .jsonl).",
    )
    parser.add_argument(
        "--out",
        default="submission.csv",
        metavar="OUTPUT_CSV",
        help="Output CSV file path (default: submission.csv).",
    )
    return parser.parse_args()


def load_jd(jd_path: Path):
    """Read and parse the JD .docx file into a JobDescription object."""
    from app.services.dynamic_evaluation_service import get_dynamic_evaluation_service

    if not jd_path.exists():
        logger.error(f"JD file not found: {jd_path}")
        sys.exit(1)

    if jd_path.suffix.lower() != ".docx":
        logger.error(f"JD file must be a .docx file, got: {jd_path.suffix}")
        sys.exit(1)

    logger.info(f"Loading Job Description from: {jd_path}")
    jd_bytes = jd_path.read_bytes()

    service = get_dynamic_evaluation_service()
    try:
        jd = service.process_jd(jd_bytes)
        logger.info(f"JD loaded successfully ({jd.paragraph_count} paragraphs)")
        return jd
    except Exception as e:
        logger.error(f"Failed to parse JD: {e}")
        sys.exit(1)


def load_candidates(candidates_path: Path):
    """Read and parse the candidates file into a list of Candidate objects."""
    from app.services.dynamic_evaluation_service import get_dynamic_evaluation_service

    if not candidates_path.exists():
        logger.error(f"Candidates file not found: {candidates_path}")
        sys.exit(1)

    suffix = candidates_path.suffix.lower()
    if suffix not in (".json", ".jsonl"):
        logger.warning(
            f"Unexpected file extension '{suffix}'. Will attempt to auto-detect format."
        )

    logger.info(f"Loading candidates from: {candidates_path}")
    raw_text = candidates_path.read_text(encoding="utf-8")

    service = get_dynamic_evaluation_service()
    try:
        # Let the service auto-detect json array vs jsonl
        candidates = service.parse_candidates(raw_text)
        logger.info(f"Parsed {len(candidates)} candidates successfully.")
        return candidates
    except ValueError as e:
        logger.error(f"Failed to parse candidates: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Internal helpers for chunked evaluation
# ---------------------------------------------------------------------------

def _chunk_list(lst: list, chunk_size: int) -> List[list]:
    """Split *lst* into successive sublists of at most *chunk_size* elements."""
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def _evaluate_chunk(chunk_idx: int, jd, chunk: list, evaluator, behavioral_evaluator):
    """
    Evaluate a single chunk of candidates.

    Runs relevance evaluation and behavioral evaluation **concurrently** (the
    same strategy as the original single-batch path) and returns the combined
    results as a tuple so the caller can merge them later.

    This function is designed to be called from a ThreadPoolExecutor worker.
    FastEmbed (ONNX runtime) releases the GIL during inference, so multiple
    threads calling this function simultaneously achieve real parallelism for
    the embedding step.

    Args:
        chunk_idx:           Zero-based chunk index (used only for logging).
        jd:                  Parsed JobDescription object shared across chunks.
        chunk:               Subset of Candidate objects to evaluate.
        evaluator:           Shared RelevanceEvaluator instance (thread-safe read).
        behavioral_evaluator: Shared BehavioralEvaluator instance (thread-safe read).

    Returns:
        (chunk_idx, relevance_result, behavioral_result)
    """
    import asyncio

    logger.info(
        f"[Chunk {chunk_idx}] Evaluating {len(chunk)} candidates "
        f"(relevance + behavioral in parallel)..."
    )

    async def _parallel():
        rel, beh = await asyncio.gather(
            asyncio.to_thread(evaluator.evaluate_all_candidates, jd, chunk),
            asyncio.to_thread(behavioral_evaluator.evaluate_all_candidates, chunk),
        )
        return rel, beh

    # Each worker thread gets its own event loop.
    loop = asyncio.new_event_loop()
    try:
        relevance_result, behavioral_result = loop.run_until_complete(_parallel())
    finally:
        loop.close()

    logger.info(
        f"[Chunk {chunk_idx}] Done — "
        f"relevance: {len(relevance_result.evaluations)}, "
        f"behavioral: {len(behavioral_result)}"
    )
    return chunk_idx, relevance_result, behavioral_result


def run_ranking(jd, candidates):
    """
    Run the full ranking pipeline:
      1. Relevance evaluation (vector similarity + rule-based)
      2. Behavioral evaluation (career trajectory heuristics)
      3. Consolidation + ranking (Redrob signals, honeypot detection)

    For small candidate sets (<= CHUNK_THRESHOLD) the evaluation runs as a
    single batch (original behaviour).  For larger sets the candidates are
    split into chunks of CHUNK_SIZE and evaluated concurrently across up to
    MAX_CONCURRENT_CHUNKS worker threads.  All partial results are merged
    before the single consolidation + ranking step.

    Returns a RankingResponse object.
    """
    import asyncio
    from app.services.relevance_evaluator import RelevanceEvaluator
    from app.services.behavioral_evaluator import BehavioralEvaluator
    from app.services.ranking_service import ConsolidationService

    evaluator = RelevanceEvaluator()
    behavioral_evaluator = BehavioralEvaluator()
    consolidation_service = ConsolidationService()

    total = len(candidates)
    logger.info(f"Starting ranking pipeline for {total} candidates...")

    # ------------------------------------------------------------------
    # Decide: single-batch path vs chunked parallel path
    # ------------------------------------------------------------------
    if total <= CHUNK_THRESHOLD:
        # ---- Original single-batch path (small candidate set) --------
        logger.info(
            f"Candidate count ({total}) <= CHUNK_THRESHOLD ({CHUNK_THRESHOLD}). "
            "Running single-batch evaluation (relevance + behavioral in parallel)."
        )

        async def _run_parallel():
            rel, beh = await asyncio.gather(
                asyncio.to_thread(evaluator.evaluate_all_candidates, jd, candidates),
                asyncio.to_thread(behavioral_evaluator.evaluate_all_candidates, candidates),
            )
            return rel, beh

        relevance_result, behavioral_result = asyncio.run(_run_parallel())
        all_relevance_evals = relevance_result.evaluations
        all_behavioral_evals = behavioral_result
        # Reuse the jd_summary produced by the single batch
        jd_summary = relevance_result.jd_summary

    else:
        # ---- Chunked parallel path (large candidate set) -------------
        chunks = _chunk_list(candidates, CHUNK_SIZE)
        num_chunks = len(chunks)
        workers = min(MAX_CONCURRENT_CHUNKS, num_chunks)

        logger.info(
            f"Candidate count ({total}) > CHUNK_THRESHOLD ({CHUNK_THRESHOLD}). "
            f"Splitting into {num_chunks} chunk(s) of up to {CHUNK_SIZE} candidates each. "
            f"Dispatching up to {workers} chunk(s) concurrently "
            f"(MAX_CONCURRENT_CHUNKS={MAX_CONCURRENT_CHUNKS})."
        )

        # Collect partial results indexed by chunk_idx so we can merge in order.
        partial_results = [None] * num_chunks  # (relevance_result, behavioral_result)
        jd_summary = None  # Will be taken from the first completed chunk.

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(
                    _evaluate_chunk,
                    chunk_idx,
                    jd,
                    chunk,
                    evaluator,
                    behavioral_evaluator,
                ): chunk_idx
                for chunk_idx, chunk in enumerate(chunks)
            }

            completed = 0
            for future in as_completed(future_to_idx):
                chunk_idx, rel_result, beh_result = future.result()
                partial_results[chunk_idx] = (rel_result, beh_result)
                completed += 1
                logger.info(
                    f"Progress: {completed}/{num_chunks} chunks complete "
                    f"({completed * CHUNK_SIZE:,} / {total:,} candidates processed)."
                )
                # Capture jd_summary from whichever chunk finishes first.
                if jd_summary is None:
                    jd_summary = rel_result.jd_summary

        # Merge all partial relevance + behavioral evaluations.
        logger.info("All chunks complete. Merging partial evaluation results...")
        all_relevance_evals = []
        all_behavioral_evals = []
        for rel_result, beh_result in partial_results:
            all_relevance_evals.extend(rel_result.evaluations)
            all_behavioral_evals.extend(beh_result)

        logger.info(
            f"Merge complete — "
            f"relevance: {len(all_relevance_evals)} evals, "
            f"behavioral: {len(all_behavioral_evals)} evals."
        )

    # ------------------------------------------------------------------
    # Step 3: Single consolidation + ranking pass (always)
    # ------------------------------------------------------------------
    logger.info("Consolidating and ranking candidates...")
    ranking_response = consolidation_service.consolidate_and_rank(
        candidates=candidates,
        relevance_evals=all_relevance_evals,
        behavioral_evals=all_behavioral_evals,
        jd_summary=jd_summary,
    )

    logger.info(
        f"Ranking complete. {ranking_response.total_candidates} candidates ranked."
    )
    return ranking_response


def write_csv(ranking_response, out_path: Path):
    """
    Write ranked results to a submission-spec-compliant CSV.

    Columns: candidate_id, rank, score, reasoning
    - rank:      integer 1-N (1 = best match)
    - score:     monotonically non-increasing float (0.2000-0.9920)
    - reasoning: 1-2 sentence justification string

    Only the top TOP_N_RESULTS (100) candidates are written, regardless of
    how many were evaluated.  This matches the submission spec of exactly
    1 header row + 100 data rows.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = ranking_response.results[:TOP_N_RESULTS]
    total_evaluated = ranking_response.total_candidates
    logger.info(
        f"Writing top {len(rows)} of {total_evaluated} evaluated candidates to: {out_path}"
    )

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        # Header row (required by spec)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for ranked in rows:
            writer.writerow(
                [
                    ranked.candidate_id,
                    ranked.rank,
                    ranked.final_score,
                    ranked.justification.strip().replace("\n", " "),
                ]
            )

    logger.info(f"CSV written successfully -> {out_path}")


def main():
    args = parse_args()

    jd_path = Path(args.jd).resolve()
    candidates_path = Path(args.candidates).resolve()
    out_path = Path(args.out).resolve()

    logger.info("=" * 60)
    logger.info("Intelligent Candidate Discovery -- Offline Ranker")
    logger.info("=" * 60)
    logger.info(f"  JD file        : {jd_path}")
    logger.info(f"  Candidates file: {candidates_path}")
    logger.info(f"  Output CSV     : {out_path}")
    logger.info("=" * 60)

    # Load inputs
    jd = load_jd(jd_path)
    candidates = load_candidates(candidates_path)

    # Run the full ranking pipeline
    ranking_response = run_ranking(jd, candidates)

    # Write output CSV
    write_csv(ranking_response, out_path)

    logger.info("")
    logger.info("Done! Summary:")
    logger.info(f"  Total candidates ranked : {ranking_response.total_candidates}")
    jd_summary_preview = ranking_response.job_description_summary[:120]
    logger.info(f"  JD summary              : {jd_summary_preview}...")
    logger.info(f"  Output file             : {out_path}")


if __name__ == "__main__":
    main()
