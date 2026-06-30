"""
Relevance Evaluation Service

Combines vector similarity (embeddings-based) and skill/experience analysis.
Key responsibilities:
- Generate embeddings for JD and candidate profiles
- Compute cosine similarity scores (0.0-1.0)
- Analyze skills & experience fit (0-100)
- Aggregate into a combined relevance score (0.0-1.0)
"""

from typing import List, Tuple, Dict, Any, Optional
import os
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

from app.models.candidate import Candidate
from app.models.jd import JobDescription
from app.models.evaluation import (
    VectorSimilarityResult,
    RelevanceEvaluation,
    EvaluationResult,
)
from app.services.embedding_service import get_embedding_service
from app.constants import (
    TITLE_MATCH_MAX_SCORE,
    TITLE_KEYWORD_MULTIPLIER,
    SCORING_VALUES,
    EXPERIENCE_TARGET_MIN,
    EXPERIENCE_TARGET_MAX,
    EXPERIENCE_BASE_SCORE,
    FUZZY_THRESHOLD,
    VECTOR_SIMILARITY_WEIGHT,
    RULE_BASED_WEIGHT,
)
# Validate weights sum to approximately 1.0
_total_weight = VECTOR_SIMILARITY_WEIGHT + RULE_BASED_WEIGHT
if not (0.99 <= _total_weight <= 1.01):
    raise ValueError(
        f"Weightages must sum to 1.0. "
        f"Got {VECTOR_SIMILARITY_WEIGHT} + {RULE_BASED_WEIGHT} = {_total_weight}"
    )

class RuleBasedSkillMatcher:
    """
    Rule-based skill matching using exact and fuzzy matching with Levenshtein distance.
    """
    
    FUZZY_THRESHOLD = FUZZY_THRESHOLD  # 75% similarity for fuzzy match
    
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> float:
        """
        Calculate similarity ratio using SequenceMatcher (0.0-1.0).
        Higher values indicate better matches.
        """
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
    
    @staticmethod
    def find_skill_match(candidate_skill: str, jd_skills: List[str]) -> Tuple[bool, float, str]:
        """
        Find a match for candidate_skill in jd_skills using exact and fuzzy matching.
        
        Returns:
            (is_match: bool, score: float 0-1, matched_skill: str or "")
        """
        candidate_skill_lower = candidate_skill.lower().strip()
        
        # Exact match (highest priority)
        for jd_skill in jd_skills:
            if candidate_skill_lower == jd_skill.lower().strip():
                return True, 1.0, jd_skill
        
        # Fuzzy match with Levenshtein distance
        best_match = None
        best_score = 0.0
        
        for jd_skill in jd_skills:
            similarity = RuleBasedSkillMatcher.levenshtein_distance(
                candidate_skill, jd_skill
            )
            if similarity > best_score:
                best_score = similarity
                best_match = jd_skill
        
        # Return fuzzy match if above threshold
        if best_score >= RuleBasedSkillMatcher.FUZZY_THRESHOLD:
            return True, best_score, best_match
        
        return False, best_score, ""
    
    @staticmethod
    def calculate_skill_match_score(
        candidate_skills: List[str],
        jd_required_skills: List[str]
    ) -> Tuple[float, List[str], List[str]]:
        """
        Calculate Skill Match Score (0-100) based on percentage of JD required skills found.
        
        Args:
            candidate_skills: List of candidate's skills
            jd_required_skills: List of required skills from JD
        
        Returns:
            (match_score: 0-100, matched_skills: List, unmatched_skills: List)
        """
        if not jd_required_skills:
            return 100.0, candidate_skills, []
        
        matched = []
        unmatched = []
        
        for jd_skill in jd_required_skills:
            is_match, score, matched_skill = RuleBasedSkillMatcher.find_skill_match(
                jd_skill, candidate_skills
            )
            if is_match:
                matched.append(f"{jd_skill} (matched: {matched_skill}, {score:.2f})")
            else:
                unmatched.append(jd_skill)
        
        # Calculate percentage match
        match_percentage = (len(matched) / len(jd_required_skills)) * 100
        
        return match_percentage, matched, unmatched


class RelevanceEvaluator:
    """
    Evaluates candidate-to-JD relevance using FastEmbed embeddings and rule-based skill analysis.
    
    Combines:
    - Embedding-based semantic similarity (FastEmbed with cosine similarity)
    - Rule-based skill matching (exact + fuzzy string matching)
    - Experience level scoring with penalties
    - Job title relevance scoring
    - Behavioral signal analysis
    """

    def __init__(self):
        """
        Initialize evaluator using FastEmbed.
        """
        self.embedding_service = get_embedding_service()
        self.skill_matcher = RuleBasedSkillMatcher()

    def generate_jd_embedding(self, jd: JobDescription) -> np.ndarray:
        """Generate embedding for job description using FastEmbed."""
        return self.embedding_service.embed_text(jd.raw_text)

    def build_candidate_profile_text(self, candidate: Candidate) -> str:
        """Construct concatenated profile string for a candidate."""
        profile_text = f"{candidate.profile.headline} {candidate.profile.summary}"
        for job in candidate.career_history:
            profile_text += f" {job.title} {job.description} {job.company}"
        skills_text = " ".join([s.name for s in candidate.skills])
        profile_text += f" Skills: {skills_text}"
        return profile_text

    def generate_candidate_embedding(self, candidate: Candidate) -> np.ndarray:
        """Generate embedding for candidate profile using FastEmbed."""
        profile_text = self.build_candidate_profile_text(candidate)
        return self.embedding_service.embed_text(profile_text)

    def compute_vector_similarity(
        self, jd: JobDescription, candidate: Candidate
    ) -> VectorSimilarityResult:
        """Compute cosine similarity between JD and candidate embeddings using FastEmbed."""
        jd_embedding = self.generate_jd_embedding(jd)
        candidate_embedding = self.generate_candidate_embedding(candidate)

        # Normalize embeddings for cosine similarity
        similarity = self.embedding_service.cosine_similarity(jd_embedding, candidate_embedding)

        # Clamp to [0, 1]
        similarity = float(np.clip(similarity, 0.0, 1.0))

        return VectorSimilarityResult(cosine_similarity=similarity)

    def extract_jd_requirements(self, jd: JobDescription) -> Dict[str, Any]:
        """
        Dynamically extract requirements from JD text using structural parsing.
        """
        jd_text = jd.raw_text
        jd_lower = jd_text.lower()
        
        requirements = {
            "must_have": [],
            "nice_to_have": [],
            "disqualifiers": [],
            "title_keywords": [],
            "experience": {
                "min": EXPERIENCE_TARGET_MIN,
                "max": EXPERIENCE_TARGET_MAX
            }
        }

        # 1. Experience extraction
        exp_match = re.search(r'(\d+)\s*[-–to]+\s*(\d+)\s*years?', jd_lower)
        if exp_match:
            requirements["experience"]["min"] = float(exp_match.group(1))
            requirements["experience"]["max"] = float(exp_match.group(2))

        # 2. Title keyword extraction (Look at first non-empty line, skip generic headers)
        lines = [l.strip() for s in jd_text.split('\n') if (l := s.strip())]
        if lines:
            for line in lines[:3]: # Check first 3 lines
                line_lower = line.lower().strip(" :")
                if line_lower in ["overview", "scope", "about the role", "introduction", "job description"]:
                    continue
                # Clean the title
                clean_title = re.sub(r'[^\w\s]', ' ', line)
                words = [w for w in clean_title.split() if len(w) > 1 and w.lower() not in ["the", "and", "for", "with", "of", "at", "in", "to", "company"]]
                if words:
                    requirements["title_keywords"] = words
                    break

        # 3. Structural Keyword Extraction
        current_section = "must_have"
        
        for line in lines:
            line_lower = line.lower().strip(" :")
            
            # Detect section headers
            if any(h in line_lower for h in ["look", "requirement", "qualif", "skill", "must have"]):
                current_section = "must_have"
                continue
            elif any(h in line_lower for h in ["nice to have", "preferred", "bonus", "plus", "advantage"]):
                current_section = "nice_to_have"
                continue
            elif any(h in line_lower for h in ["disqualifiers", "not looking for"]):
                current_section = "disqualifiers"
                continue
                
            # Extract bullet points OR lines that look like requirements (start with cap or verb)
            # Match: symbol OR capitalized word OR specific keywords like 'Strong', 'Experience', 'Familiarity'
            is_requirement = re.match(r'^[-*•]\s+', line.strip()) or \
                            (re.match(r'^[A-Z]', line.strip()) and len(line.split()) > 3) or \
                            any(line.strip().startswith(w) for w in ["Strong", "Experience", "Knowledge", "Ability", "Familiarity"])
            
            if is_requirement:
                # Clean the item
                clean_item = re.sub(r'^[-*•]\s+', '', line.strip())
                # Split by delimiters
                skills = re.split(r',|\sand\s|\sor\s|\/', clean_item)
                for skill in skills:
                    skill_clean = skill.strip(" .;")
                    # Keep short phrases (1-5 words)
                    if 0 < len(skill_clean.split()) <= 5:
                        requirements[current_section].append(skill_clean)

        # Deduplicate and remove empties
        requirements["must_have"] = list(set([s for s in requirements["must_have"] if s]))
        requirements["nice_to_have"] = list(set([s for s in requirements["nice_to_have"] if s]))
        requirements["disqualifiers"] = list(set([s for s in requirements["disqualifiers"] if s]))

        return requirements

    def calculate_experience_score(
        self, 
        candidate_years: float, 
        target_min: float = EXPERIENCE_TARGET_MIN, 
        target_max: float = EXPERIENCE_TARGET_MAX
    ) -> Tuple[float, str]:
        """
        Calculate experience score with penalties for gaps.
        """
        if target_min <= candidate_years <= target_max:
            return 10.0, f"[MATCH] Target experience range ({candidate_years:.1f} yrs)"
        elif target_min - 1 <= candidate_years < target_min:
            return 5.0, f"[APPROX] Slightly below range ({candidate_years:.1f} yrs)"
        elif target_max < candidate_years <= target_max + 3:
            return 7.0, f"[APPROX] Slightly above range ({candidate_years:.1f} yrs)"
        elif candidate_years > target_max + 3:
            return 3.0, f"[APPROX] Much above range ({candidate_years:.1f} yrs)"
        else:
            penalty = min(-15.0, -(target_min - candidate_years) * 5)
            return penalty, f"[FAIL] Below range ({candidate_years:.1f} yrs)"

    def calculate_title_relevance(
        self, 
        candidate_current_title: str, 
        candidate_history_titles: List[str], 
        target_keywords: List[str]
    ) -> Tuple[float, str]:
        """
        Compare job titles using dynamically extracted keywords.
        """
        if not target_keywords:
            return 0.0, "[INFO] No title keywords found in JD"

        titles_to_check = [candidate_current_title] + candidate_history_titles[:3]
        
        best_score = 0.0
        best_match = ""
        
        for title in titles_to_check:
            title_lower = title.lower()
            
            # Check for target keywords
            keyword_matches = sum(1 for kw in target_keywords if kw in title_lower)
            if keyword_matches > 0:
                score = min(TITLE_MATCH_MAX_SCORE, keyword_matches * TITLE_KEYWORD_MULTIPLIER)
                if score > best_score:
                    best_score = score
                    best_match = title
        
        if best_score > 0:
            return best_score, f"[MATCH] Relevant title: '{best_match}'"
        else:
            return 0.0, f"[INFO] Non-technical title: '{candidate_current_title}'"

    def calculate_rule_based_score(self, jd: JobDescription, candidate: Candidate, requirements: Optional[Dict[str, Any]] = None) -> int:
        """
        Calculate rule-based skill and experience match score (0-100).
        """
        # Dynamically extract requirements for this specific JD if not pre-provided
        if requirements is None:
            requirements = self.extract_jd_requirements(jd)
        
        score = EXPERIENCE_BASE_SCORE
        factors = []
        
        candidate_skills_names = [s.name for s in candidate.skills]
        career_desc = " ".join([job.description.lower() for job in candidate.career_history])
        career_titles = [job.title for job in candidate.career_history]
        all_text = f"{candidate.profile.summary.lower()} {career_desc} {' '.join([s.lower() for s in candidate_skills_names])}".lower()
        
        # ============ SKILL MATCHING ============
        jd_all_skills = requirements["must_have"] + requirements["nice_to_have"]
        
        if jd_all_skills:
            skill_match_score, matched, unmatched = self.skill_matcher.calculate_skill_match_score(
                candidate_skills_names, jd_all_skills
            )
            
            skill_contribution = (skill_match_score / 100.0) * SCORING_VALUES["skill_match_max_contribution"]
            score += skill_contribution
            
            if skill_match_score >= 80:
                factors.append((f"High skill match ({skill_match_score:.0f}%)", skill_contribution))
            elif skill_match_score >= 50:
                factors.append((f"Medium skill match ({skill_match_score:.0f}%)", skill_contribution))
            else:
                factors.append((f"Low skill match ({skill_match_score:.0f}%)", skill_contribution))
        
        # ============ EXPERIENCE MATCHING ============
        exp_score, exp_explanation = self.calculate_experience_score(
            candidate.profile.years_of_experience,
            target_min=requirements["experience"]["min"],
            target_max=requirements["experience"]["max"]
        )
        score += exp_score
        factors.append((exp_explanation, exp_score))
        
        # ============ TITLE RELEVANCE ============
        title_score, title_explanation = self.calculate_title_relevance(
            candidate.profile.current_title,
            career_titles,
            requirements["title_keywords"]
        )
        score += title_score
        if title_score > 0:
            factors.append((title_explanation, title_score))
        
        # ============ KEYWORD ANALYSIS ============
        # Score must-haves
        for mw in requirements["must_have"]:
            if mw in all_text:
                score += SCORING_VALUES["must_have_bonus"]
                factors.append((f"[MATCH] Found must-have: {mw}", SCORING_VALUES["must_have_bonus"]))
        
        # Score nice-to-haves
        for nh in requirements["nice_to_have"]:
            if nh in all_text:
                score += SCORING_VALUES["opensource_bonus"]
                factors.append((f"[MATCH] Found nice-to-have: {nh}", SCORING_VALUES["opensource_bonus"]))

        # Check disqualifiers
        for dq in requirements["disqualifiers"]:
            if dq in all_text or any(dq in t.lower() for t in career_titles):
                score += SCORING_VALUES["cv_only_penalty"]
                factors.append((f"[FAIL] Found disqualifier: {dq}", SCORING_VALUES["cv_only_penalty"]))

        # ============ BEHAVIORAL SIGNALS ============
        if candidate.redrob_signals.open_to_work_flag:
            score += SCORING_VALUES["open_to_work_bonus"]
            factors.append(("[MATCH] Open to work", SCORING_VALUES["open_to_work_bonus"]))
        else:
            score += SCORING_VALUES["not_actively_looking_penalty"]
            factors.append(("[FAIL] Not actively looking", SCORING_VALUES["not_actively_looking_penalty"]))
        
        if candidate.redrob_signals.recruiter_response_rate > 0.7:
            score += SCORING_VALUES["high_engagement_bonus"]
            factors.append((f"[MATCH] High engagement ({candidate.redrob_signals.recruiter_response_rate:.0%})", SCORING_VALUES["high_engagement_bonus"]))
        elif candidate.redrob_signals.recruiter_response_rate < 0.3:
            score += SCORING_VALUES["low_engagement_penalty"]
            factors.append((f"[FAIL] Low engagement ({candidate.redrob_signals.recruiter_response_rate:.0%})", SCORING_VALUES["low_engagement_penalty"]))
        
        if candidate.redrob_signals.github_activity_score >= 60:
            score += SCORING_VALUES["strong_github_bonus"]
            factors.append((f"[MATCH] Strong GitHub ({candidate.redrob_signals.github_activity_score}/100)", SCORING_VALUES["strong_github_bonus"]))
        
        notice = candidate.redrob_signals.notice_period_days
        if notice <= 30:
            score += SCORING_VALUES["quick_notice_bonus"]
            factors.append((f"[MATCH] Quick notice ({notice} days)", SCORING_VALUES["quick_notice_bonus"]))
        elif notice > 90:
            score += SCORING_VALUES["long_notice_penalty"]
            factors.append((f"[FAIL] Long notice ({notice} days)", SCORING_VALUES["long_notice_penalty"]))
        
        if candidate.profile.country.lower() == "india":
            score += SCORING_VALUES["india_based_bonus"]
            factors.append(("[MATCH] India-based", SCORING_VALUES["india_based_bonus"]))
        
        return int(np.clip(score, 0, 100))

    def evaluate_candidate(
        self, jd: JobDescription, candidate: Candidate, requirements: Optional[Dict[str, Any]] = None, vector_sim: Optional[VectorSimilarityResult] = None
    ) -> RelevanceEvaluation:
        """Evaluate a single candidate against the JD."""
        if vector_sim is None:
            vector_sim = self.compute_vector_similarity(jd, candidate)

        rule_based_score = self.calculate_rule_based_score(jd, candidate, requirements=requirements)

        rule_normalized = rule_based_score / 100.0
        combined_score = VECTOR_SIMILARITY_WEIGHT * vector_sim.cosine_similarity + RULE_BASED_WEIGHT * rule_normalized

        return RelevanceEvaluation(
            candidate_id=candidate.candidate_id,
            vector_similarity=vector_sim,
            rule_based_score=rule_based_score,
            combined_score=float(combined_score),
        )

    def _compute_rule_factors(self, jd: JobDescription, candidate: Candidate) -> str:
        """
        Compute top factors for rule-based scoring (for display in tests).
        """
        requirements = self.extract_jd_requirements(jd)
        factors = []
        
        candidate_skills_names = [s.name for s in candidate.skills]
        jd_all_skills = requirements["must_have"] + requirements["nice_to_have"]
        
        if jd_all_skills:
            skill_match_score, matched, unmatched = self.skill_matcher.calculate_skill_match_score(
                candidate_skills_names, jd_all_skills
            )
            if skill_match_score >= 80:
                factors.append(f"High skill match ({skill_match_score:.0f}%)")
            elif skill_match_score >= 50:
                factors.append(f"Medium skill match ({skill_match_score:.0f}%)")
            else:
                factors.append(f"Low skill match ({skill_match_score:.0f}%)")
            
            if unmatched:
                factors.append(f"Missing: {', '.join(unmatched[:2])}")
        
        exp_score, exp_explanation = self.calculate_experience_score(
            candidate.profile.years_of_experience,
            target_min=requirements["experience"]["min"],
            target_max=requirements["experience"]["max"]
        )
        factors.append(exp_explanation)
        
        return "; ".join(factors[:3])

    def evaluate_all_candidates(
        self, jd: JobDescription, candidates: List[Candidate]
    ) -> EvaluationResult:
        """Evaluate all candidates against the JD with high-performance batching and vectorization."""
        logger.info(f"Starting relevance evaluation for {len(candidates)} candidates.")
        requirements = self.extract_jd_requirements(jd)
        jd_embedding = self.generate_jd_embedding(jd)
        jd_norm = np.linalg.norm(jd_embedding)

        logger.info("Building profile texts and running FastEmbed batch embedding...")
        profile_texts = [self.build_candidate_profile_text(c) for c in candidates]
        candidate_embeddings = self.embedding_service.embed_texts(profile_texts)

        logger.info("Computing matrix cosine similarities...")
        cand_matrix = np.array(candidate_embeddings, dtype=np.float32)
        cand_norms = np.linalg.norm(cand_matrix, axis=1)
        
        valid_mask = (cand_norms > 0) & (jd_norm > 0)
        similarities = np.zeros(len(candidates), dtype=np.float32)
        if jd_norm > 0:
            similarities[valid_mask] = np.dot(cand_matrix[valid_mask], jd_embedding) / (cand_norms[valid_mask] * jd_norm)
        similarities = np.clip(similarities, 0.0, 1.0)

        logger.info("Calculating rule-based scores in parallel...")
        evaluations = [None] * len(candidates)

        def _score_candidate(idx: int, candidate: Candidate) -> tuple:
            sim_val = float(similarities[idx])
            vector_sim = VectorSimilarityResult(cosine_similarity=sim_val)
            return idx, self.evaluate_candidate(
                jd, candidate, requirements=requirements, vector_sim=vector_sim
            )

        with ThreadPoolExecutor(max_workers=min(8, len(candidates))) as executor:
            futures = {executor.submit(_score_candidate, i, c): i for i, c in enumerate(candidates)}
            for future in as_completed(futures):
                idx, result = future.result()
                evaluations[idx] = result

        evaluations.sort(key=lambda x: x.combined_score, reverse=True)

        title_str = " ".join(requirements["title_keywords"]).title()
        jd_summary = f"""
Job Role: {title_str}
Experience: {requirements['experience']['min']}-{requirements['experience']['max']} years
Must-have: {", ".join(requirements['must_have'][:5])}
Nice-to-have: {", ".join(requirements['nice_to_have'][:5])}
        """.strip()

        logger.info("Relevance evaluation batch completed successfully.")
        return EvaluationResult(
            evaluations=evaluations,
            jd_summary=jd_summary,
        )
