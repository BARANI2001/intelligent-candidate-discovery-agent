"""
Consolidation and Ranking Service (Redrob Signals Integration)

Combines heuristic skill-match (Relevance) score and algorithmic Behavioral score,
then applies Redrob engagement/availability multipliers to determine final ranking.
"""

from typing import List, Dict, Any
from datetime import datetime
import logging
import numpy as np

logger = logging.getLogger(__name__)

from app.models.candidate import Candidate
from app.models.evaluation import RelevanceEvaluation, BehavioralEvaluation
from app.models.ranking import RankedCandidate, RankingResponse
from app.services.honeypot_detector import HoneypotDetector


class ConsolidationService:
    """
    Consolidates scores from Step 3 (Relevance) and Step 4 (Behavioral) with Redrob platform signals.
    Honeypot candidates (impossible profiles) are detected and hard-floored to score ≤ 5.0.
    """

    _honeypot_detector = HoneypotDetector()

    AI_CORE_KEYWORDS = {
        "ai", "ml", "machine learning", "deep learning", "nlp", "natural language processing",
        "computer vision", "image classification", "object detection", "speech recognition", "tts",
        "fine-tuning", "fine-tuning llms", "llm", "llms", "large language models", "generative ai",
        "transformers", "huggingface", "pytorch", "tensorflow", "keras", "scikit-learn",
        "weights & biases", "lora", "qlora", "rag", "retrieval augmented generation", "embeddings",
        "vector database", "milvus", "qdrant", "pinecone", "chromadb", "faiss", "weaviate",
        "gans", "reinforcement learning", "statistical modeling", "feature engineering", "data science",
        "neural networks", "bentoml", "mlops", "model deployment", "spark mllib", "apache beam"
    }

    @classmethod
    def count_ai_core_skills(cls, candidate: Candidate) -> int:
        """Count unique AI/ML core skills present in candidate profile."""
        count = 0
        if not candidate.skills:
            return 0
        for skill in candidate.skills:
            s_name = skill.name.strip().lower()
            if s_name in cls.AI_CORE_KEYWORDS or any(kw in s_name for kw in ["ai", "ml", "neural", "llm", "model", "vision", "speech", "embedding", "tuning"]):
                count += 1
        return count

    @staticmethod
    def _is_inactive_over_6_months(last_active_date_str: str) -> bool:
        """Check if last active date is older than 180 days from reference date."""
        try:
            # Try parsing ISO standard YYYY-MM-DD
            last_active = datetime.strptime(last_active_date_str[:10], "%Y-%m-%d")
            # For hackathon dataset consistency, compare against a reference date or current time
            # Assume dataset baseline reference is around mid-2024 or current date
            now = datetime.now()
            days_inactive = (now - last_active).days
            return days_inactive > 180
        except Exception:
            return False

    def consolidate_and_rank(
        self,
        candidates: List[Candidate],
        relevance_evals: List[RelevanceEvaluation],
        behavioral_evals: List[BehavioralEvaluation],
        jd_summary: str
    ) -> RankingResponse:
        """Combine scores, apply multipliers, generate justifications, and rank."""
        logger.info(f"Consolidating scores and ranking {len(candidates)} candidates...")
        
        # Map evaluations by candidate_id for O(1) lookup
        rel_map = {e.candidate_id: e for e in relevance_evals}
        beh_map = {e.candidate_id: e for e in behavioral_evals}

        ranked_list: List[RankedCandidate] = []

        for cand in candidates:
            cid = cand.candidate_id
            rel = rel_map.get(cid)
            beh = beh_map.get(cid)

            # Convert combined relevance score (0.0-1.0) to 0-100 scale, or use rule_based_score
            rel_score = float(rel.combined_score * 100.0) if rel else 50.0
            beh_score = float(beh.composite_behavioral_score) if beh else 50.0

            # 1. Base Score formula: (Relevance * 0.6) + (Behavioral * 0.4)
            base_score = (rel_score * 0.6) + (beh_score * 0.4)

            # 2. Redrob Signal Modifiers
            signals = cand.redrob_signals
            multipliers: Dict[str, Any] = {}
            justification_parts = []

            # Base breakdown
            justification_parts.append(f"Base Score {base_score:.1f} (60% Relevance [{rel_score:.1f}] + 40% Behavior [{beh_score:.1f}])")

            # A. Availability Multiplier
            is_inactive = self._is_inactive_over_6_months(signals.last_active_date)
            if not signals.open_to_work_flag or is_inactive:
                avail_mult = 0.5
                multipliers["availability_multiplier"] = 0.5
                reason = "Not open to work" if not signals.open_to_work_flag else f"Inactive since {signals.last_active_date}"
                justification_parts.append(f"Severe Availability Penalty 0.5x ({reason})")
            else:
                avail_mult = 1.0
                multipliers["availability_multiplier"] = 1.0

            # B. Engagement Multiplier
            avg_engagement = (signals.recruiter_response_rate + signals.interview_completion_rate) / 2.0
            if avg_engagement >= 0.8:
                eng_mult = 1.1
                multipliers["engagement_multiplier"] = 1.1
                justification_parts.append(f"High Engagement Boost 1.1x (Response rate: {signals.recruiter_response_rate:.0%}, Interview completion: {signals.interview_completion_rate:.0%})")
            elif avg_engagement <= 0.3:
                eng_mult = 0.8
                multipliers["engagement_multiplier"] = 0.8
                justification_parts.append(f"Low Engagement Penalty 0.8x (Response rate: {signals.recruiter_response_rate:.0%})")
            else:
                eng_mult = 1.0
                multipliers["engagement_multiplier"] = 1.0

            # C. Profile Quality Boost
            quality_boost = 0.0
            avg_assessment = sum(signals.skill_assessment_scores.values()) / max(1, len(signals.skill_assessment_scores)) if signals.skill_assessment_scores else 0.0
            if signals.profile_completeness_score >= 85.0 and avg_assessment >= 75.0:
                quality_boost = 5.0
                multipliers["profile_quality_boost"] = +5.0
                justification_parts.append(f"Profile Quality Bonus +5.0 pts (Completeness: {signals.profile_completeness_score:.0f}%, Avg Assessment: {avg_assessment:.0f})")
            else:
                multipliers["profile_quality_boost"] = 0.0

            # D. Flight Risk Penalty
            flight_risk_penalty = 0.0
            if 0.0 <= signals.offer_acceptance_rate < 0.3:
                flight_risk_penalty = -10.0
                multipliers["flight_risk_penalty"] = -10.0
                justification_parts.append(f"Flight Risk Penalty -10.0 pts (Low historical offer acceptance rate: {signals.offer_acceptance_rate:.0%})")
            else:
                multipliers["flight_risk_penalty"] = 0.0

            # Compute Final Score
            final_score = (base_score * avail_mult * eng_mult) + quality_boost + flight_risk_penalty
            final_score = float(np.clip(final_score, 0.0, 100.0))

            # E. Honeypot Detection — hard floor to ≤ 5.0 for confirmed honeypots
            honeypot = self._honeypot_detector.detect(cand)
            if honeypot.is_honeypot:
                final_score = min(final_score, 5.0)
                multipliers["honeypot_penalty"] = True
                justification_parts.append(
                    f"HONEYPOT DETECTED (score={honeypot.honeypot_score:.2f}): "
                    + " | ".join(honeypot.flags)
                )
            elif honeypot.honeypot_score >= 0.25:
                # Suspicious — soft penalty only
                soft_penalty = honeypot.honeypot_score * 10.0
                final_score = max(0.0, final_score - soft_penalty)
                multipliers["honeypot_suspicion_penalty"] = round(-soft_penalty, 2)
                justification_parts.append(
                    f"Suspicious profile penalty -{soft_penalty:.1f} pts: "
                    + " | ".join(honeypot.flags)
                )
            final_score = float(np.clip(final_score, 0.0, 100.0))

            # Build submission-quality reasoning (1-2 sentences, spec-compliant)
            reasoning = self._build_reasoning(
                cand, rel, beh, honeypot, rel_score, beh_score, final_score, jd_summary
            )

            # Keep a compact internal justification for API debug use
            ai_skills_count = self.count_ai_core_skills(cand)
            title = cand.profile.current_title or "Professional"
            yrs = cand.profile.years_of_experience
            resp_rate = signals.recruiter_response_rate
            benchmark_reasoning = f"{title} with {yrs:.1f} yrs; {ai_skills_count} AI core skills; response rate {resp_rate:.2f}."

            ranked_list.append(
                RankedCandidate(
                    candidate_id=cid,
                    rank=1,  # Placeholder to be set after sorting
                    final_score=round(final_score, 2),
                    base_score=round(base_score, 2),
                    relevance_score=round(rel_score, 2),
                    behavioral_score=round(beh_score, 2),
                    applied_multipliers=multipliers,
                    justification=benchmark_reasoning,
                    reasoning=reasoning,
                    relevance_justification=f"Vector Sim: {rel.vector_similarity.cosine_similarity:.2f}, Rule Score: {rel.rule_based_score}/100" if rel else None,
                    behavioral_summary="; ".join(justification_parts) + f". (Behavioral: {beh.summary})" if beh else "; ".join(justification_parts) + "."
                )
            )

        # Sort descending by final_score
        ranked_list.sort(key=lambda x: x.final_score, reverse=True)

        # Assign integer ranks starting from 1 and normalize final_score to monotonic hackathon scale
        for idx, item in enumerate(ranked_list, 1):
            item.rank = idx
            # Linearly scale score starting from 0.9920 down by 0.0080 per rank (min 0.2000)
            item.final_score = round(max(0.2000, 0.9920 - (idx - 1) * 0.0080), 4)

        logger.info("Consolidation and ranking complete.")
        return RankingResponse(
            job_description_summary=jd_summary,
            total_candidates=len(ranked_list),
            results=ranked_list
        )

    # ------------------------------------------------------------------ #
    #  Submission Reasoning Builder                                        #
    # ------------------------------------------------------------------ #

    def _build_reasoning(
        self,
        cand,
        rel,
        beh,
        honeypot,
        rel_score: float,
        beh_score: float,
        final_score: float,
        jd_summary: str,
    ) -> str:
        """
        Build a concise 1-2 sentence reasoning string for the submission CSV.

        Rules (from spec Stage 4 manual review):
          - References specific facts: title, years, company, named skills
          - Connects to JD requirements
          - Honestly acknowledges concerns where applicable
          - No hallucination: only uses data actually in the profile
          - Tone matches rank (top candidates = positive; bottom = honest limitations)
        """
        p = cand.profile
        signals = cand.redrob_signals

        # === Part 1: Core identity + skill match ===
        title = p.current_title or "Candidate"
        yrs = p.years_of_experience
        company = p.current_company or ""
        company_str = f" at {company}" if company else ""

        # Top 3 skills by proficiency weight (expert > advanced > intermediate > beginner)
        proficiency_rank = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
        top_skills = sorted(
            cand.skills,
            key=lambda s: (proficiency_rank.get(s.proficiency, 0), s.endorsements),
            reverse=True
        )[:3]
        skills_str = ", ".join(s.name for s in top_skills) if top_skills else "general skills"

        # Rule-based score label
        if rel_score >= 75:
            match_label = f"strong JD skill match ({rel_score:.0f}/100)"
        elif rel_score >= 50:
            match_label = f"moderate JD skill match ({rel_score:.0f}/100)"
        else:
            match_label = f"limited JD skill match ({rel_score:.0f}/100)"

        sentence1 = (
            f"{title} with {yrs:.1f} yrs{company_str}; "
            f"top skills: {skills_str}; {match_label}."
        )

        # === Part 2: Signals + concerns ===
        concerns = []
        positives = []

        # Engagement
        if signals.recruiter_response_rate >= 0.8:
            positives.append(f"high recruiter response rate ({signals.recruiter_response_rate:.0%})")
        elif signals.recruiter_response_rate < 0.3:
            concerns.append(f"low response rate ({signals.recruiter_response_rate:.0%})")

        # Availability
        if not signals.open_to_work_flag:
            concerns.append("not currently open to work")

        # Notice period
        notice = signals.notice_period_days
        if notice > 90:
            concerns.append(f"long notice period ({notice}d)")
        elif notice <= 15:
            positives.append(f"immediate availability ({notice}d notice)")

        # GitHub
        if signals.github_activity_score >= 70:
            positives.append(f"strong GitHub activity ({signals.github_activity_score:.0f}/100)")

        # Behavioral trajectory
        if beh and beh.demotions_detected > 0:
            concerns.append(f"{beh.demotions_detected} career demotion(s) detected")

        # Honeypot flag
        if honeypot.is_honeypot:
            concerns.append("profile has impossible consistency (honeypot flagged)")

        # Compose sentence 2
        if positives and not concerns:
            sentence2 = "Positives: " + "; ".join(positives) + "."
        elif concerns and not positives:
            sentence2 = "Concerns: " + "; ".join(concerns) + "."
        elif positives and concerns:
            sentence2 = (
                "Positives: " + "; ".join(positives) +
                ". Concerns: " + "; ".join(concerns) + "."
            )
        else:
            # Neutral — use behavioral score
            sentence2 = f"Behavioral trajectory score: {beh_score:.0f}/100."

        return f"{sentence1} {sentence2}"
