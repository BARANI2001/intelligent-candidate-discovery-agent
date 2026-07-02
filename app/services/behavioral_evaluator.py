"""
Behavioral and Career Trajectory Analysis Service (Algorithmic Heuristics)

Evaluates candidate career history without LLMs by applying deterministic Python logic:
- Tenure and Job-Hopping Analysis: Total duration, avg tenure per company, penalties for frequent changes.
- Career Progression Analysis: Mapping job title seniority tiers, checking promotions vs demotions.
- Composite Behavioral Score: 0-100 normalized score.
"""

from typing import List, Dict, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

logger = logging.getLogger(__name__)

from app.models.candidate import Candidate, CareerHistory
from app.models.evaluation import BehavioralEvaluation


class BehavioralEvaluator:
    """
    Evaluates behavioral patterns and career trajectory purely through code heuristics.
    """

    SENIORITY_HIERARCHY: Dict[str, int] = {
        "intern": 1,
        "trainee": 1,
        "junior": 2,
        "associate": 2,
        "engineer": 3,
        "developer": 3,
        "analyst": 3,
        "consultant": 3,
        "specialist": 3,
        "senior": 4,
        "sr": 4,
        "lead": 5,
        "principal": 6,
        "staff": 6,
        "architect": 6,
        "manager": 7,
        "mgr": 7,
        "head": 8,
        "director": 9,
        "vp": 10,
        "vice president": 10,
        "chief": 11,
        "cto": 11,
        "ceo": 11,
        "cpo": 11,
        "founder": 11,
        "co-founder": 11
    }

    def _map_title_to_tier(self, title: str) -> int:
        """Map a job title string to a seniority hierarchy tier integer (1-11). Default is 3."""
        title_lower = title.lower()
        best_tier = 3  # Default mid-level professional
        for keyword, tier in self.SENIORITY_HIERARCHY.items():
            # Match word boundary or exact substring
            if f" {keyword} " in f" {title_lower} " or title_lower.startswith(f"{keyword} ") or title_lower.endswith(f" {keyword}") or title_lower == keyword:
                if tier > best_tier or (best_tier == 3 and tier != 3):
                    best_tier = tier
        return best_tier

    def evaluate_candidate(self, candidate: Candidate) -> BehavioralEvaluation:
        """Analyze a candidate's career history and return a BehavioralEvaluation."""
        history = candidate.career_history
        if not history:
            return BehavioralEvaluation(
                candidate_id=candidate.candidate_id,
                tenure_score=50.0,
                progression_score=50.0,
                composite_behavioral_score=50.0,
                total_duration_months=int(candidate.profile.years_of_experience * 12),
                avg_tenure_months=float(candidate.profile.years_of_experience * 12),
                job_hopping_penalty_applied=False,
                demotions_detected=0,
                summary="No detailed career history provided; assigned baseline scores."
            )

        # 1. Tenure and Job-Hopping Analysis
        total_months = sum(max(0, role.duration_months) for role in history)
        if total_months == 0 and candidate.profile.years_of_experience > 0:
            total_months = int(candidate.profile.years_of_experience * 12)

        # Group by company to get avg tenure per company
        companies = set(role.company.lower().strip() for role in history)
        num_companies = max(1, len(companies))
        avg_tenure = total_months / num_companies

        tenure_score = 60.0  # Base tenure score
        job_hopping_penalty = False
        tenure_notes = []

        if avg_tenure > 36:
            tenure_score += 30.0
            tenure_notes.append(f"Excellent average tenure ({avg_tenure:.1f}m/company)")
        elif avg_tenure >= 24:
            tenure_score += 20.0
            tenure_notes.append(f"Solid average tenure ({avg_tenure:.1f}m/company)")
        elif avg_tenure >= 18:
            tenure_score += 10.0
            tenure_notes.append(f"Moderate average tenure ({avg_tenure:.1f}m/company)")
        else:
            # Progressive penalties for < 18 months
            shortfall = 18.0 - avg_tenure
            penalty = min(40.0, shortfall * 2.5)
            tenure_score -= penalty
            job_hopping_penalty = True
            tenure_notes.append(f"Job-hopping penalty applied (avg tenure {avg_tenure:.1f}m < 18m)")

        tenure_score = float(np.clip(tenure_score, 0.0, 100.0))

        # 2. Career Progression Analysis
        # Sort chronologically (oldest first) based on start_date
        sorted_roles = sorted(history, key=lambda r: r.start_date)
        tiers = [self._map_title_to_tier(role.title) for role in sorted_roles]

        progression_score = 60.0
        demotions = 0
        downward_fluctuations = 0
        progression_notes = []

        if len(tiers) >= 2:
            increases = sum(1 for i in range(len(tiers)-1) if tiers[i+1] > tiers[i])
            for i in range(len(tiers)-1):
                if tiers[i+1] < tiers[i]:
                    demotions += 1
                    if tiers[i] - tiers[i+1] >= 2:
                        downward_fluctuations += 1

            if increases > 0 and demotions == 0:
                progression_score += min(35.0, increases * 15.0)
                progression_notes.append(f"Consistent upward progression ({increases} promotions)")
            elif tiers[-1] >= 5 and demotions == 0:
                progression_score += 20.0
                progression_notes.append(f"Sustained senior seniority level (Tier {tiers[-1]})")

            if demotions > 0:
                penalty = demotions * 15.0
                progression_score -= penalty
                progression_notes.append(f"Demotion penalty applied ({demotions} tier decrease detected)")
            if downward_fluctuations > 0:
                progression_score -= downward_fluctuations * 10.0
                progression_notes.append(f"Additional fluctuation penalty ({downward_fluctuations} sharp drops)")
        else:
            if tiers[0] >= 4:
                progression_score += 15.0
                progression_notes.append(f"Single senior role (Tier {tiers[0]})")
            else:
                progression_notes.append(f"Single role (Tier {tiers[0]})")

        progression_score = float(np.clip(progression_score, 0.0, 100.0))

        # 3. Composite Behavioral Score (50% tenure, 50% progression)
        composite = float((tenure_score * 0.5) + (progression_score * 0.5))
        summary_text = "; ".join(tenure_notes + progression_notes)

        return BehavioralEvaluation(
            candidate_id=candidate.candidate_id,
            tenure_score=round(tenure_score, 2),
            progression_score=round(progression_score, 2),
            composite_behavioral_score=round(composite, 2),
            total_duration_months=total_months,
            avg_tenure_months=round(avg_tenure, 2),
            job_hopping_penalty_applied=job_hopping_penalty,
            demotions_detected=demotions,
            summary=summary_text
        )

    def evaluate_all_candidates(self, candidates: List[Candidate]) -> List[BehavioralEvaluation]:
        """Evaluate behavioral metrics for all candidates in parallel."""
        logger.info(f"Starting behavioral evaluation for {len(candidates)} candidates.")
        results = [None] * len(candidates)
        with ThreadPoolExecutor(max_workers=min(8, len(candidates))) as executor:
            futures = {executor.submit(self.evaluate_candidate, c): i for i, c in enumerate(candidates)}
            for future in as_completed(futures):
                i = futures[future]
                results[i] = future.result()
        logger.info("Behavioral evaluation complete.")
        return results
