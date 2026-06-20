from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict

CompanySize = Literal[
    "1-10",
    "11-50",
    "51-200",
    "201-500",
    "501-1000",
    "1001-5000",
    "5001-10000",
    "10001+",
]


class Profile(BaseModel):
    anonymized_name: str = Field(
        description="An anonymized identifier for the candidate."
    )
    headline: str = Field(description="Professional headline displayed on the profile.")
    summary: str
    location: str
    country: str
    years_of_experience: float = Field(
        ge=0, le=50, description="Total years of professional experience."
    )
    current_title: str
    current_company: str
    current_company_size: CompanySize
    current_industry: str


class CareerHistory(BaseModel):
    company: str
    title: str
    start_date: str
    end_date: Optional[str]
    duration_months: int = Field(ge=0)
    is_current: bool
    industry: str
    company_size: CompanySize
    description: str


class Education(BaseModel):
    institution: str
    degree: str
    field_of_study: str
    start_year: int = Field(ge=1970, le=2030)
    end_year: int = Field(ge=1970, le=2035)
    grade: Optional[str] = None
    tier: Literal["tier_1", "tier_2", "tier_3", "tier_4", "unknown"]


class Skill(BaseModel):
    name: str
    proficiency: Literal["beginner", "intermediate", "advanced", "expert"]
    endorsements: int = Field(ge=0)
    duration_months: Optional[int] = Field(default=None, ge=0)


class Certification(BaseModel):
    name: str
    issuer: str
    year: int


class Language(BaseModel):
    language: str
    proficiency: Literal["basic", "conversational", "professional", "native"]


class SalaryRange(BaseModel):
    """
    Represents a candidate's expected salary range in INR lakhs per annum (LPA).

    This model is used exclusively within RedrobSignals as the
    `expected_salary_range_inr_lpa` field. Currency and compensation
    frequency are implied by the parent field name and are therefore
    not stored as separate attributes.
    """

    min: float = Field(
        ge=0, description="Minimum expected salary in INR lakhs per annum."
    )
    max: float = Field(
        ge=0, description="Maximum expected salary in INR lakhs per annum."
    )


class RedrobSignals(BaseModel):
    """
    The 23 behavioral/platform-activity signals.
    """

    profile_completeness_score: float = Field(ge=0, le=100)
    signup_date: str
    last_active_date: str
    open_to_work_flag: bool
    profile_views_received_30d: int = Field(ge=0)
    applications_submitted_30d: int = Field(ge=0)
    recruiter_response_rate: float = Field(ge=0, le=1)
    avg_response_time_hours: float = Field(ge=0)
    skill_assessment_scores: Dict[str, float]
    connection_count: int = Field(ge=0)
    endorsements_received: int = Field(ge=0)
    notice_period_days: int = Field(ge=0, le=180)
    expected_salary_range_inr_lpa: SalaryRange
    preferred_work_mode: Literal["remote", "hybrid", "onsite", "flexible"]
    willing_to_relocate: bool
    github_activity_score: float = Field(
        ge=-1,
        le=100,
        description="GitHub activity score (0-100); use -1 when no GitHub profile is connected.",
    )
    search_appearance_30d: int = Field(ge=0)
    saved_by_recruiters_30d: int = Field(ge=0)
    interview_completion_rate: float = Field(ge=0, le=1)
    offer_acceptance_rate: float = Field(
        ge=-1,
        le=1,
        description="Offer acceptance rate (0-1); use -1 when the candidate has never received an offer.",
    )
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool


class Candidate(BaseModel):
    """
    Complete candidate profile including work history, education,
    skills, certifications, languages, and platform activity signals.
    """

    candidate_id: str = Field(
        pattern=r"^CAND_[0-9]{7}$",
        description="Candidate ID in the format CAND_ followed by exactly 7 digits (e.g., CAND_0001234).",
    )
    profile: Profile
    career_history: List[CareerHistory] = Field(
        min_length=1,
        max_length=10,
        description="Employment history ordered from most recent to oldest.",
    )
    education: List[Education] = Field(default_factory=list, max_length=5)
    skills: List[Skill] = Field(default_factory=list)
    certifications: Optional[List[Certification]] = []
    languages: Optional[List[Language]] = []
    redrob_signals: RedrobSignals
