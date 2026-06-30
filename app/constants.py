"""
Shared constants and evaluation configuration for the application.
"""

import os

# Job Title Relevance Constants

TITLE_MATCH_MAX_SCORE = 10.0
TITLE_KEYWORD_MULTIPLIER = 3.0


# Scoring Values (Unified map for easy tuning)

SCORING_VALUES = {
    "skill_match_max_contribution": 25.0,
    "must_have_bonus": 5.0,
    "must_have_count_multiplier": 2.0,
    "learning_to_rank_bonus": 5.0,
    "hrtech_bonus": 3.0,
    "opensource_bonus": 3.0,
    "pure_research_penalty": -20.0,
    "consulting_only_penalty": -15.0,
    "cv_only_penalty": -15.0,
    "open_to_work_bonus": 2.0,
    "not_actively_looking_penalty": -3.0,
    "high_engagement_bonus": 3.0,
    "low_engagement_penalty": -3.0,
    "strong_github_bonus": 3.0,
    "quick_notice_bonus": 2.0,
    "long_notice_penalty": -3.0,
    "india_based_bonus": 2.0,
}


# Experience Scoring Constants

EXPERIENCE_TARGET_MIN = 5.0
EXPERIENCE_TARGET_MAX = 9.0
EXPERIENCE_BASE_SCORE = 40


# Evaluation Logic Constants

FUZZY_THRESHOLD = 0.75  # 75% similarity for fuzzy match

# Weights for combined score calculation
VECTOR_SIMILARITY_WEIGHT = float(os.getenv("VECTOR_SIMILARITY_WEIGHT", "0.4"))
RULE_BASED_WEIGHT = float(os.getenv("RULE_BASED_WEIGHT", "0.6"))
