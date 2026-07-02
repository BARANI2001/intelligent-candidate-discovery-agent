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
"""

import argparse
import csv
import sys
import logging
from pathlib import Path

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


def run_ranking(jd, candidates):
    """
    Run the full ranking pipeline:
      1. Relevance evaluation (vector similarity + rule-based)
      2. Behavioral evaluation (career trajectory heuristics)
      3. Consolidation + ranking (Redrob signals, honeypot detection)
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

    # Step 1 + 2: Run relevance and behavioral evaluation concurrently
    async def _run_parallel():
        logger.info("Running relevance + behavioral evaluation in parallel...")
        relevance_result, behavioral_result = await asyncio.gather(
            asyncio.to_thread(evaluator.evaluate_all_candidates, jd, candidates),
            asyncio.to_thread(behavioral_evaluator.evaluate_all_candidates, candidates),
        )
        return relevance_result, behavioral_result

    relevance_result, behavioral_result = asyncio.run(_run_parallel())
    logger.info(
        f"Evaluation complete. Relevance: {len(relevance_result.evaluations)} evals, "
        f"Behavioral: {len(behavioral_result)} evals."
    )

    # Step 3: Consolidate and rank
    logger.info("Consolidating and ranking candidates...")
    ranking_response = consolidation_service.consolidate_and_rank(
        candidates=candidates,
        relevance_evals=relevance_result.evaluations,
        behavioral_evals=behavioral_result,
        jd_summary=relevance_result.jd_summary,
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
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = ranking_response.results
    logger.info(f"Writing {len(rows)} rows to: {out_path}")

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
