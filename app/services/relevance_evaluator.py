"""
Relevance Evaluation Service

Combines vector similarity (embeddings-based) and skill/experience analysis.
Key responsibilities:
- Generate embeddings for JD and candidate profiles
- Compute cosine similarity scores (0.0-1.0)
- Analyze skills & experience fit (0-100)
- Aggregate into a combined relevance score (0.0-1.0)

Rule-based skill matching uses:
- Exact and fuzzy keyword matching (Levenshtein distance)
- Experience level validation with penalties
- Job title relevance scoring
"""

import json
from typing import List, Tuple, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

from app.models.candidate import Candidate
from app.models.jd import JobDescription
from app.models.evaluation import (
    VectorSimilarityResult,
    RelevanceEvaluation,
    EvaluationResult,
)
from app.services.embedding_service import get_embedding_service


class MockEmbeddingModel:
    """
    Mock embedding model for development.
    In production, replace with: OpenAI text-embedding-3-small, BGE, E5, etc.
    
    Creates deterministic embeddings using word frequency weighting.
    """

    EMBEDDING_DIM = 128

    def encode(self, text: str) -> np.ndarray:
        """Generate a deterministic embedding from text."""
        import hashlib
        
        text_lower = text.lower()[:10000]
        
        # Tokenize
        words = [w.strip('.,;:!?"\'-') for w in text_lower.split() if len(w.strip('.,;:!?"\'-')) > 2]
        
        if not words:
            return np.ones(self.EMBEDDING_DIM, dtype=np.float32) / np.sqrt(self.EMBEDDING_DIM)
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Create weighted embedding
        embedding = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
        total_weight = 0
        
        # Process unique words (up to 100)
        for i, word in enumerate(list(word_freq.keys())[:100]):
            # Hash the word to get deterministic values
            word_hash = hashlib.sha256(word.encode()).digest()
            
            # Map each byte of hash to embedding dimension
            for j in range(self.EMBEDDING_DIM):
                byte_val = word_hash[j % 32]  # 32 bytes from SHA256
                # Create value in [-1, 1]
                val = (byte_val / 128.0) - 1.0
                embedding[j] += val * np.sqrt(word_freq[word])
            
            total_weight += np.sqrt(word_freq[word])
        
        # Normalize by weight
        if total_weight > 0:
            embedding = embedding / total_weight
        
        # L2 normalization
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm
        else:
            embedding = np.ones(self.EMBEDDING_DIM, dtype=np.float32) / np.sqrt(self.EMBEDDING_DIM)
        
        # Ensure finite
        embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
        return embedding.astype(np.float32)


class RuleBasedSkillMatcher:
    """
    Rule-based skill matching using exact and fuzzy matching with Levenshtein distance.
    """
    
    FUZZY_THRESHOLD = 0.75  # 75% similarity for fuzzy match
    
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

    def __init__(self, use_fastembed: bool = True):
        """
        Initialize evaluator.
        
        Args:
            use_fastembed: Use FastEmbed for embeddings (True) or mock embeddings (False)
        """
        self.embedding_service = get_embedding_service() if use_fastembed else None
        self.skill_matcher = RuleBasedSkillMatcher()
        
        # JD requirements keywords (from actual JD: Senior AI Engineer @ Redrob)
        self.jd_keywords = {
            "must_have": {
                "embeddings": ["embedding", "embeddings", "sentence-transformer", "openai embedding", "bge", "e5"],
                "retrieval": ["retrieval", "semantic search", "rag", "dense retrieval", "hybrid search"],
                "vector_db": ["pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", "faiss", "vector database"],
                "python": ["python", "pyspark", "sklearn", "numpy", "pandas"],
                "ranking": ["ranking", "ndcg", "mrr", "map", "evaluation", "relevance"],
            },
            "nice_to_have": {
                "learning_to_rank": ["learning-to-rank", "l2r", "xgboost", "neural rank", "ranknet"],
                "hrtech": ["recruiting", "hr-tech", "talent", "recruitment", "candidate matching", "job matching"],
                "distributed": ["distributed", "kafka", "spark", "mapreduce", "hadoop"],
                "opensource": ["github", "open-source", "opensource", "repository", "contrib"],
            },
            "disqualifiers": {
                "pure_research": ["research", "academic", "phd", "postdoc"],
                "recent_langchain": ["langchain", "openai api", "chatgpt wrapper"],
                "no_recent_code": ["architect", "tech lead", "manager", "no production"],
                "only_consulting": ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"],
                "cv_only": ["computer vision", "cv specialist", "speech recognition", "robotics"],
            },
        }

    def generate_jd_embedding(self, jd: JobDescription) -> np.ndarray:
        """Generate embedding for job description using FastEmbed."""
        if self.embedding_service:
            # Use FastEmbed for real embeddings
            text = jd.raw_text[:2000]
            return self.embedding_service.embed_text(text)
        else:
            # Fallback to mock embeddings
            text = jd.raw_text[:2000]
            # Use mock embedding (for backward compatibility)
            return np.random.randn(384).astype(np.float32)

    def generate_candidate_embedding(self, candidate: Candidate) -> np.ndarray:
        """Generate embedding for candidate profile using FastEmbed."""
        # Concatenate relevant fields
        profile_text = f"""
        {candidate.profile.headline}
        {candidate.profile.summary}
        """

        # Add career history
        for job in candidate.career_history[:3]:  # Recent 3 jobs
            profile_text += f" {job.title} {job.description} {job.company}"

        # Add skills
        skills_text = " ".join([s.name for s in candidate.skills[:20]])
        profile_text += f" Skills: {skills_text}"

        # Use first 2000 chars
        text = profile_text[:2000]
        
        if self.embedding_service:
            # Use FastEmbed for real embeddings
            return self.embedding_service.embed_text(text)
        else:
            # Fallback to mock embeddings
            return np.random.randn(384).astype(np.float32)

    def compute_vector_similarity(
        self, jd: JobDescription, candidate: Candidate
    ) -> VectorSimilarityResult:
        """Compute cosine similarity between JD and candidate embeddings using FastEmbed."""
        jd_embedding = self.generate_jd_embedding(jd)
        candidate_embedding = self.generate_candidate_embedding(candidate)

        # Normalize embeddings for cosine similarity
        if self.embedding_service:
            similarity = self.embedding_service.cosine_similarity(jd_embedding, candidate_embedding)
        else:
            # Fallback: use sklearn
            jd_vec = jd_embedding.reshape(1, -1)
            cand_vec = candidate_embedding.reshape(1, -1)
            similarity = float(cosine_similarity(jd_vec, cand_vec)[0][0])

        # Clamp to [0, 1]
        similarity = float(np.clip(similarity, 0.0, 1.0))

        return VectorSimilarityResult(cosine_similarity=similarity)

    def extract_jd_required_skills(self, jd: JobDescription) -> List[str]:
        """
        Extract required skills from JD text.
        Uses keyword matching from must-have and nice-to-have categories.
        """
        jd_text_lower = jd.raw_text.lower()
        extracted_skills = set()
        
        # Collect all keywords from must-have and nice-to-have
        all_categories = list(self.jd_keywords["must_have"].values()) + \
                         list(self.jd_keywords["nice_to_have"].values())
        
        for category_keywords in all_categories:
            for keyword in category_keywords:
                if keyword in jd_text_lower:
                    extracted_skills.add(keyword)
        
        return sorted(list(extracted_skills))

    def calculate_experience_score(
        self, candidate_years: float, target_min: float = 5.0, target_max: float = 9.0
    ) -> Tuple[float, str]:
        """
        Calculate experience score with penalties for gaps.
        
        Scoring:
        - Perfect match (target_min <= years <= target_max): +10
        - Slightly below (target_min - 1 <= years < target_min): +5
        - Slightly above (target_max < years <= target_max + 3): +7
        - Significantly above (years > target_max + 3): +3
        - Significantly below (years < target_min - 1): -15 (penalty)
        
        Returns:
            (score: float, explanation: str)
        """
        if target_min <= candidate_years <= target_max:
            return 10.0, f"✓ Target experience range ({candidate_years:.1f} yrs)"
        elif target_min - 1 <= candidate_years < target_min:
            return 5.0, f"~ Slightly below range ({candidate_years:.1f} yrs)"
        elif target_max < candidate_years <= target_max + 3:
            return 7.0, f"~ Slightly above range ({candidate_years:.1f} yrs)"
        elif candidate_years > target_max + 3:
            return 3.0, f"~ Much above range ({candidate_years:.1f} yrs)"
        else:
            penalty = min(-15.0, -(target_min - candidate_years) * 5)  # Proportional penalty
            return penalty, f"✗ Below range ({candidate_years:.1f} yrs)"

    def calculate_title_relevance(
        self, candidate_current_title: str, candidate_history_titles: List[str], jd_text: str
    ) -> Tuple[float, str]:
        """
        Compare job titles using fuzzy matching.
        
        Returns:
            (score: 0-15, explanation: str)
        """
        target_keywords = ["senior", "engineer", "ai", "ml", "machine learning", "lead", "architect"]
        
        titles_to_check = [candidate_current_title] + candidate_history_titles[:3]
        
        best_score = 0.0
        best_match = ""
        
        for title in titles_to_check:
            title_lower = title.lower()
            
            # Check for target keywords
            keyword_matches = sum(1 for kw in target_keywords if kw in title_lower)
            if keyword_matches > 0:
                best_score = max(best_score, min(10.0, keyword_matches * 3))
                best_match = title
        
        if best_score > 0:
            return best_score, f"✓ Relevant title: '{best_match}'"
        else:
            return 0.0, f"? Non-technical title: '{candidate_current_title}'"

    def calculate_rule_based_score(self, jd: JobDescription, candidate: Candidate) -> int:
        """
        Calculate rule-based skill and experience match score (0-100).
        
        Scoring approach:
        1. Extract required skills from JD
        2. Calculate Skill Match Score using exact and fuzzy matching (0-100)
        3. Calculate experience score with penalties for gaps
        4. Calculate title relevance score
        5. Check for disqualifiers
        6. Consider behavioral signals
        
        Returns score 0-100.
        """
        score = 40  # Base score
        factors = []  # Track scoring factors
        
        # Prepare candidate text for analysis
        candidate_skills_names = [s.name for s in candidate.skills]
        skills_text = " ".join([s.lower() for s in candidate_skills_names])
        career_desc = " ".join([job.description.lower() for job in candidate.career_history])
        career_titles = [job.title for job in candidate.career_history]
        all_text = f"{candidate.profile.summary.lower()} {career_desc} {skills_text}".lower()
        
        # ============ SKILL MATCHING ============
        jd_required_skills = self.extract_jd_required_skills(jd)
        
        if jd_required_skills:
            skill_match_score, matched, unmatched = self.skill_matcher.calculate_skill_match_score(
                candidate_skills_names, jd_required_skills
            )
            
            # Apply skill match to overall score
            skill_contribution = (skill_match_score / 100.0) * 25  # Max 25 points
            score += skill_contribution
            
            if skill_match_score >= 80:
                factors.append((f"High skill match ({skill_match_score:.0f}%)", skill_contribution))
            elif skill_match_score >= 50:
                factors.append((f"Medium skill match ({skill_match_score:.0f}%)", skill_contribution))
            else:
                factors.append((f"Low skill match ({skill_match_score:.0f}%)", skill_contribution))
            
            if unmatched:
                factors.append((f"Missing: {', '.join(unmatched[:2])}", 0))
        
        # ============ EXPERIENCE MATCHING ============
        exp_score, exp_explanation = self.calculate_experience_score(
            candidate.profile.years_of_experience
        )
        score += exp_score
        factors.append((exp_explanation, exp_score))
        
        # ============ TITLE RELEVANCE ============
        title_score, title_explanation = self.calculate_title_relevance(
            candidate.profile.current_title,
            career_titles,
            jd.raw_text
        )
        score += title_score
        if title_score > 0:
            factors.append((title_explanation, title_score))
        
        # ============ MUST-HAVE KEYWORDS ============
        must_have_matches = 0
        
        if any(keyword in all_text for keyword in self.jd_keywords["must_have"]["embeddings"]):
            must_have_matches += 1
            factors.append(("✓ Embeddings/retrieval experience", 5))
        
        if any(keyword in all_text for keyword in self.jd_keywords["must_have"]["vector_db"]):
            must_have_matches += 1
            factors.append(("✓ Vector database experience", 5))
        
        if any(keyword in all_text for keyword in self.jd_keywords["must_have"]["ranking"]):
            must_have_matches += 1
            factors.append(("✓ Ranking/evaluation experience", 5))
        
        score += must_have_matches * 2
        
        # ============ NICE-TO-HAVE ============
        
        if any(keyword in all_text for keyword in self.jd_keywords["nice_to_have"]["learning_to_rank"]):
            score += 5
            factors.append(("✓ Learning-to-rank experience", 5))
        
        if any(keyword in all_text for keyword in self.jd_keywords["nice_to_have"]["hrtech"]):
            score += 3
            factors.append(("✓ HR-tech domain experience", 3))
        
        if any(keyword in all_text for keyword in self.jd_keywords["nice_to_have"]["opensource"]):
            score += 3
            factors.append(("✓ Open-source contributions", 3))
        
        # ============ DISQUALIFIERS ============
        
        pure_research = any(
            keyword in career_titles for keyword in ["research", "academic", "phd"]
        ) and "production" not in career_desc
        
        if pure_research and candidate.profile.years_of_experience > 4:
            score -= 20
            factors.append(("✗ Pure research background", -20))
        
        only_consulting = all(
            company.lower().strip() in ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "mindtree"]
            for company in [job.company.lower() for job in candidate.career_history]
        ) and len(candidate.career_history) >= 2
        
        if only_consulting:
            score -= 15
            factors.append(("✗ Consulting firm only", -15))
        
        cv_only = any(
            keyword in career_desc for keyword in self.jd_keywords["disqualifiers"]["cv_only"]
        ) and "nlp" not in career_desc and "retrieval" not in career_desc
        
        if cv_only:
            score -= 15
            factors.append(("✗ CV/robotics primary", -15))
        
        # ============ BEHAVIORAL SIGNALS ============
        
        if candidate.redrob_signals.open_to_work_flag:
            score += 2
            factors.append(("✓ Open to work", 2))
        else:
            score -= 3
            factors.append(("✗ Not actively looking", -3))
        
        if candidate.redrob_signals.recruiter_response_rate > 0.7:
            score += 3
            factors.append((f"✓ High engagement ({candidate.redrob_signals.recruiter_response_rate:.0%})", 3))
        elif candidate.redrob_signals.recruiter_response_rate < 0.3:
            score -= 3
            factors.append((f"✗ Low engagement ({candidate.redrob_signals.recruiter_response_rate:.0%})", -3))
        
        if candidate.redrob_signals.github_activity_score >= 60:
            score += 3
            factors.append((f"✓ Strong GitHub ({candidate.redrob_signals.github_activity_score}/100)", 3))
        
        notice = candidate.redrob_signals.notice_period_days
        if notice <= 30:
            score += 2
            factors.append((f"✓ Quick notice ({notice} days)", 2))
        elif notice > 90:
            score -= 3
            factors.append((f"✗ Long notice ({notice} days)", -3))
        
        if candidate.profile.country.lower() == "india":
            score += 2
            factors.append(("✓ India-based", 2))
        
        # Clamp to 0-100 range
        return int(np.clip(score, 0, 100))

    def evaluate_candidate(
        self, jd: JobDescription, candidate: Candidate
    ) -> RelevanceEvaluation:
        """Evaluate a single candidate against the JD."""
        # Get vector similarity
        vector_sim = self.compute_vector_similarity(jd, candidate)

        # Get rule-based score
        rule_based_score = self.calculate_rule_based_score(jd, candidate)

        # Combine scores (normalize rule-based score to 0-1, then blend)
        rule_normalized = rule_based_score / 100.0
        combined_score = 0.4 * vector_sim.cosine_similarity + 0.6 * rule_normalized

        return RelevanceEvaluation(
            candidate_id=candidate.candidate_id,
            vector_similarity=vector_sim,
            rule_based_score=rule_based_score,
            combined_score=float(combined_score),
        )

    def _compute_rule_factors(self, jd: JobDescription, candidate: Candidate) -> str:
        """
        Compute top factors for rule-based scoring (for display in tests).
        Returns a string with top 3 factors.
        """
        factors = []
        score = 40
        
        candidate_skills_names = [s.name for s in candidate.skills]
        career_desc = " ".join([job.description.lower() for job in candidate.career_history])
        career_titles = [job.title for job in candidate.career_history]
        all_text = f"{candidate.profile.summary.lower()} {career_desc}".lower()
        skills_text = " ".join([s.lower() for s in candidate_skills_names]).lower()
        
        jd_required_skills = self.extract_jd_required_skills(jd)
        
        if jd_required_skills:
            skill_match_score, matched, unmatched = self.skill_matcher.calculate_skill_match_score(
                candidate_skills_names, jd_required_skills
            )
            skill_contribution = (skill_match_score / 100.0) * 25
            score += skill_contribution
            
            if skill_match_score >= 80:
                factors.append(f"High skill match ({skill_match_score:.0f}%)")
            elif skill_match_score >= 50:
                factors.append(f"Medium skill match ({skill_match_score:.0f}%)")
            else:
                factors.append(f"Low skill match ({skill_match_score:.0f}%)")
            
            if unmatched:
                factors.append(f"Missing: {', '.join(unmatched[:2])}")
        
        exp_score, exp_explanation = self.calculate_experience_score(candidate.profile.years_of_experience)
        score += exp_score
        factors.append(exp_explanation)
        
        title_score, title_explanation = self.calculate_title_relevance(
            candidate.profile.current_title, career_titles, jd.raw_text
        )
        score += title_score
        if title_score > 0:
            factors.append(title_explanation)
        
        if any(keyword in all_text for keyword in self.jd_keywords["must_have"]["embeddings"]):
            factors.append("✓ Embeddings experience")
        
        if any(keyword in all_text for keyword in self.jd_keywords["must_have"]["vector_db"]):
            factors.append("✓ Vector database experience")
        
        if any(keyword in all_text for keyword in self.jd_keywords["nice_to_have"]["learning_to_rank"]):
            factors.append("✓ Learning-to-rank experience")
        
        if any(keyword in all_text for keyword in self.jd_keywords["nice_to_have"]["hrtech"]):
            factors.append("✓ HR-tech domain experience")
        
        return "; ".join(factors[:3])
    
    def _get_factor_score(self, factor: str) -> float:
        """Get score contribution for a factor string."""
        if "High skill match" in factor:
            return 25.0
        elif "Medium skill match" in factor:
            return 12.5
        elif "Low skill match" in factor:
            return 6.25
        elif "Missing" in factor:
            return 0.0
        elif "Target experience" in factor:
            return 10.0
        elif "Slightly below" in factor:
            return 5.0
        elif "Slightly above" in factor:
            return 7.0
        elif "Much above" in factor:
            return 3.0
        elif "Below range" in factor:
            return -15.0
        elif "Relevant title" in factor:
            return 10.0
        elif "Non-technical" in factor:
            return 0.0
        elif "Embeddings experience" in factor:
            return 5.0
        elif "Vector database experience" in factor:
            return 5.0
        elif "Ranking experience" in factor:
            return 5.0
        elif "Learning-to-rank experience" in factor:
            return 5.0
        elif "HR-tech domain experience" in factor:
            return 3.0
        elif "Open-source contributions" in factor:
            return 3.0
        elif "Pure research background" in factor:
            return -20.0
        elif "Consulting firm" in factor:
            return -15.0
        elif "CV/robotics primary" in factor:
            return -15.0
        elif "Open to work" in factor:
            return 2.0
        elif "Not actively looking" in factor:
            return -3.0
        elif "High engagement" in factor:
            return 3.0
        elif "Low engagement" in factor:
            return -3.0
        elif "Strong GitHub" in factor:
            return 3.0
        elif "Quick notice" in factor:
            return 2.0
        elif "Long notice" in factor:
            return -3.0
        elif "India-based" in factor:
            return 2.0
        return 0.0

    def evaluate_all_candidates(
        self, jd: JobDescription, candidates: List[Candidate]
    ) -> EvaluationResult:
        """Evaluate all candidates against the JD."""
        evaluations = []
        for candidate in candidates:
            eval_result = self.evaluate_candidate(jd, candidate)
            evaluations.append(eval_result)

        # Sort by combined_score descending
        evaluations.sort(key=lambda x: x.combined_score, reverse=True)

        # Extract JD summary from actual JD requirements
        jd_summary = """
        Senior AI Engineer (Founding Team) @ Redrob
        
        Must-have: Production embeddings/retrieval systems, Vector databases, Strong Python, Ranking evaluation frameworks.
        Nice-to-have: Learning-to-rank models, HR-tech experience, Distributed systems.
        Disqualifiers: Pure research background, Recent LangChain-only projects, No production code in 18 months, Only consulting firm experience.
        
        Location: India (Pune/Noida preferred, Tier-1 cities welcome). Notice: <30 days preferred.
        """.strip()

        return EvaluationResult(
            evaluations=evaluations,
            jd_summary=jd_summary,
        )
