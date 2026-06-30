"""
Honeypot Detection Service

Detects candidates with subtly impossible profiles using internal consistency checks.
No external data required — all checks are derived from the candidate JSON itself.

Honeypot signals (per hackathon spec examples + generalizations):
  1. Expert/Advanced skill with 0 duration_months used
  2. Too many expert skills relative to total career months
  3. Career duration mismatch: claimed years_of_experience vs sum(career_history.duration_months)
  4. Impossible role duration: stated duration_months doesn't match (end_date - start_date)
  5. Future dates: start_date or end_date after today
  6. Impossible dates: end_date < start_date within a role
  7. Overlapping jobs: two roles at different companies with overlapping date ranges
  8. Current role has an end_date (contradicts is_current=True)
  9. Education timeline impossible: end_year < start_year
 10. Salary inversion: expected_salary min > max
"""

import logging
from datetime import date, datetime
from typing import List, Dict, Any, Tuple

from app.models.candidate import Candidate

logger = logging.getLogger(__name__)

TODAY = date.today()
# Treat dataset as frozen around competition date (mid-2026).
# Any date beyond this is "future" for a candidate profile.
DATASET_CUTOFF = date(2026, 7, 1)


def _parse_date(date_str: str | None) -> date | None:
    """Parse ISO date string YYYY-MM-DD. Returns None on failure."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


class HoneypotDetection:
    """Result of honeypot analysis for a single candidate."""

    def __init__(self, candidate_id: str):
        self.candidate_id = candidate_id
        self.flags: List[str] = []
        self.honeypot_score: float = 0.0  # 0.0 = clean, 1.0 = definite honeypot
        self.is_honeypot: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "is_honeypot": self.is_honeypot,
            "honeypot_score": round(self.honeypot_score, 4),
            "flags": self.flags,
        }


class HoneypotDetector:
    """
    Applies a set of deterministic internal-consistency checks to candidate profiles
    and returns a HoneypotDetection result with a score and explanatory flags.

    Score thresholds:
      >= 0.50 → is_honeypot = True  (hard disqualification in ranking)
      0.25–0.49 → suspicious (soft penalty only)
      < 0.25 → clean
    """

    # Weight each check contributes to honeypot_score (sum can exceed 1.0, clamped later)
    WEIGHTS = {
        "expert_zero_duration":       0.30,   # Strong signal, explicitly in spec
        "advanced_zero_duration":     0.20,   # Weaker but suspicious
        "too_many_experts":           0.20,   # ≥8 expert skills with tiny avg duration
        "experience_mismatch":        0.35,   # Claimed YoE vs career history sum
        "experience_mismatch_severe": 0.20,   # Bonus when single-role + large gap
        "impossible_role_duration":   0.25,   # duration_months ≠ date diff
        "future_start_date":          0.40,   # Start date in future
        "end_before_start":           0.50,   # End date before start date
        "current_job_has_end":        0.15,   # is_current=True but end_date set
        "overlapping_jobs":           0.20,   # Two roles at different orgs overlap
        "education_impossible":       0.20,   # end_year < start_year
        "salary_inverted":            0.15,   # min salary > max salary
    }

    HONEYPOT_THRESHOLD = 0.50
    EXPERT_SKILLS_CAP = 8  # More than this many "expert" skills is suspicious

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def detect(self, candidate: Candidate) -> HoneypotDetection:
        """Run all checks and return a HoneypotDetection for a single candidate."""
        result = HoneypotDetection(candidate.candidate_id)
        raw_score = 0.0

        raw_score += self._check_skill_proficiency_vs_duration(candidate, result)
        raw_score += self._check_experience_mismatch(candidate, result)
        raw_score += self._check_career_dates(candidate, result)
        raw_score += self._check_education(candidate, result)
        raw_score += self._check_salary(candidate, result)

        result.honeypot_score = min(1.0, raw_score)
        result.is_honeypot = result.honeypot_score >= self.HONEYPOT_THRESHOLD
        return result

    def detect_all(self, candidates: List[Candidate]) -> List[HoneypotDetection]:
        """Run detection on a list of candidates."""
        return [self.detect(c) for c in candidates]

    # ------------------------------------------------------------------ #
    #  Check: Skill Proficiency vs Duration                               #
    # ------------------------------------------------------------------ #

    def _check_skill_proficiency_vs_duration(
        self, candidate: Candidate, result: HoneypotDetection
    ) -> float:
        score = 0.0
        if not candidate.skills:
            return score

        expert_zero = 0
        advanced_zero = 0
        expert_skills = [s for s in candidate.skills if s.proficiency == "expert"]

        for skill in candidate.skills:
            dur = skill.duration_months if skill.duration_months is not None else None
            if dur is None:
                continue
            if skill.proficiency == "expert" and dur == 0:
                expert_zero += 1
            elif skill.proficiency == "advanced" and dur == 0:
                advanced_zero += 1

        if expert_zero > 0:
            result.flags.append(
                f"SKILL_EXPERT_ZERO_DURATION: {expert_zero} skill(s) marked 'expert' with 0 months used"
            )
            score += self.WEIGHTS["expert_zero_duration"]

        if advanced_zero > 0:
            result.flags.append(
                f"SKILL_ADVANCED_ZERO_DURATION: {advanced_zero} skill(s) marked 'advanced' with 0 months used"
            )
            score += self.WEIGHTS["advanced_zero_duration"]

        # Too many expert skills check
        if len(expert_skills) >= self.EXPERT_SKILLS_CAP:
            total_career_months = sum(
                max(0, r.duration_months) for r in candidate.career_history
            )
            avg_expert_duration = (
                sum(s.duration_months or 0 for s in expert_skills) / len(expert_skills)
                if expert_skills else 0
            )
            # If lots of experts but avg duration is tiny relative to career
            if total_career_months > 0 and avg_expert_duration < total_career_months * 0.1:
                result.flags.append(
                    f"SKILL_TOO_MANY_EXPERTS: {len(expert_skills)} expert skills but avg duration only "
                    f"{avg_expert_duration:.0f}m vs {total_career_months}m career"
                )
                score += self.WEIGHTS["too_many_experts"]

        return score

    # ------------------------------------------------------------------ #
    #  Check: Experience Mismatch                                         #
    # ------------------------------------------------------------------ #

    def _check_experience_mismatch(
        self, candidate: Candidate, result: HoneypotDetection
    ) -> float:
        """
        Compare profile.years_of_experience with sum of career_history.duration_months.
        Allow up to 24 months (2 years) gap — gaps happen for freelance/gaps/rounding.
        Flag if claimed experience exceeds actual career months by > 24 months.
        """
        if not candidate.career_history:
            return 0.0

        claimed_months = candidate.profile.years_of_experience * 12
        actual_months = sum(max(0, r.duration_months) for r in candidate.career_history)

        if actual_months == 0:
            return 0.0

        diff = claimed_months - actual_months
        # Tolerance: 24 months (2 years) — normal career gaps are expected
        if diff <= 24:
            return 0.0

        score = self.WEIGHTS["experience_mismatch"]
        result.flags.append(
            f"EXPERIENCE_MISMATCH: Claims {candidate.profile.years_of_experience:.1f} yrs "
            f"({claimed_months:.0f}m) but career history sums to only {actual_months}m "
            f"(gap={diff:.0f}m)"
        )

        # Amplify: single role + massive gap is near-definite honeypot
        if len(candidate.career_history) == 1 and diff > 60:
            result.flags.append(
                f"EXPERIENCE_MISMATCH_SEVERE: Only 1 role in history with {diff:.0f}m gap — "
                "strongly suggests fabricated profile"
            )
            score += self.WEIGHTS["experience_mismatch_severe"]

        return score

    # ------------------------------------------------------------------ #
    #  Check: Career Date Consistency                                     #
    # ------------------------------------------------------------------ #

    def _check_career_dates(
        self, candidate: Candidate, result: HoneypotDetection
    ) -> float:
        score = 0.0
        parsed_roles: List[Tuple[date | None, date | None, bool, str]] = []

        for role in candidate.career_history:
            start = _parse_date(role.start_date)
            end = _parse_date(role.end_date) if not role.is_current else None

            # Check: future start date
            if start and start > DATASET_CUTOFF:
                result.flags.append(
                    f"FUTURE_START_DATE: Role at '{role.company}' starts {role.start_date} (future)"
                )
                score += self.WEIGHTS["future_start_date"]

            # Check: end_date before start_date
            if start and end and end < start:
                result.flags.append(
                    f"END_BEFORE_START: Role at '{role.company}' ends {role.end_date} before start {role.start_date}"
                )
                score += self.WEIGHTS["end_before_start"]

            # Check: is_current=True but end_date present in raw JSON
            # (role.is_current=True and end_date field is not None)
            if role.is_current and role.end_date is not None:
                result.flags.append(
                    f"CURRENT_JOB_HAS_END: Role at '{role.company}' is marked current but has end_date={role.end_date}"
                )
                score += self.WEIGHTS["current_job_has_end"]

            # Check: stated duration_months vs actual date diff
            if start and end and not role.is_current:
                actual_months = (end.year - start.year) * 12 + (end.month - start.month)
                stated = role.duration_months
                diff = abs(actual_months - stated)
                if diff > 6:  # Allow 6-month rounding leeway
                    result.flags.append(
                        f"IMPOSSIBLE_DURATION: Role at '{role.company}' states {stated}m but "
                        f"date range implies {actual_months}m (diff={diff}m)"
                    )
                    score += self.WEIGHTS["impossible_role_duration"]

            if start:
                parsed_roles.append((start, end, role.is_current, role.company))

        # Check: overlapping roles at different companies
        score += self._check_overlaps(parsed_roles, result)
        return score

    def _check_overlaps(
        self,
        roles: List[Tuple[date | None, date | None, bool, str]],
        result: HoneypotDetection,
    ) -> float:
        """Flag roles at different companies with overlapping date ranges > 3 months."""
        score = 0.0
        n = len(roles)
        if n < 2:
            return score

        overlap_found = False
        for i in range(n):
            s1, e1, cur1, co1 = roles[i]
            if not s1:
                continue
            effective_e1 = e1 if e1 else DATASET_CUTOFF

            for j in range(i + 1, n):
                s2, e2, cur2, co2 = roles[j]
                if not s2:
                    continue
                effective_e2 = e2 if e2 else DATASET_CUTOFF

                # Same company — multi-title at same org is OK
                if co1.lower().strip() == co2.lower().strip():
                    continue

                # Check overlap
                overlap_start = max(s1, s2)
                overlap_end = min(effective_e1, effective_e2)
                if overlap_end > overlap_start:
                    overlap_months = (overlap_end.year - overlap_start.year) * 12 + (
                        overlap_end.month - overlap_start.month
                    )
                    if overlap_months > 3:  # >3 months of concurrent full-time jobs
                        if not overlap_found:
                            result.flags.append(
                                f"OVERLAPPING_JOBS: Roles at '{co1}' and '{co2}' overlap by "
                                f"~{overlap_months}m"
                            )
                            score += self.WEIGHTS["overlapping_jobs"]
                            overlap_found = True  # Count once per candidate

        return score

    # ------------------------------------------------------------------ #
    #  Check: Education Timeline                                          #
    # ------------------------------------------------------------------ #

    def _check_education(
        self, candidate: Candidate, result: HoneypotDetection
    ) -> float:
        score = 0.0
        for edu in candidate.education:
            if edu.end_year < edu.start_year:
                result.flags.append(
                    f"EDUCATION_IMPOSSIBLE: '{edu.institution}' end_year {edu.end_year} < start_year {edu.start_year}"
                )
                score += self.WEIGHTS["education_impossible"]
        return score

    # ------------------------------------------------------------------ #
    #  Check: Salary Inversion                                            #
    # ------------------------------------------------------------------ #

    def _check_salary(self, candidate: Candidate, result: HoneypotDetection) -> float:
        sal = candidate.redrob_signals.expected_salary_range_inr_lpa
        if sal.min > sal.max:
            result.flags.append(
                f"SALARY_INVERTED: min salary {sal.min} > max salary {sal.max} LPA"
            )
            return self.WEIGHTS["salary_inverted"]
        return 0.0
