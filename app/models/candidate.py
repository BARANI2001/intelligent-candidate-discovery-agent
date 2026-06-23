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
    anonymized_name: str = Field(description="Anonymized full name.")
    headline: str = Field(description="One-line professional headline.")
    summary: str = Field(description="Multi-sentence professional summary.")
    location: str = Field(description="City, region/state.")
    country: str
    years_of_experience: float = Field(ge=0, le=50)
    current_title: str
    current_company: str
    current_company_size: CompanySize
    current_industry: str


class CareerHistory(BaseModel):
    company: str
    title: str
    start_date: str
    end_date: Optional[str] = None
    duration_months: int = Field(ge=0)
    is_current: bool
    industry: str
    company_size: CompanySize
    description: str = Field(description="Role responsibilities and achievements.")


class Education(BaseModel):
    institution: str
    degree: str
    field_of_study: str
    start_year: int = Field(ge=1970, le=2030)
    end_year: int = Field(ge=1970, le=2035)
    grade: Optional[str] = Field(default=None, description="GPA / percentage / class.")
    tier: Literal["tier_1", "tier_2", "tier_3", "tier_4", "unknown"] = Field(description="Internal tiering for institution prestige.")


class Skill(BaseModel):
    name: str
    proficiency: Literal["beginner", "intermediate", "advanced", "expert"]
    endorsements: int = Field(ge=0)
    duration_months: Optional[int] = Field(default=None, ge=0, description="Months the candidate has used this skill")


class Certification(BaseModel):
    name: str
    issuer: str
    year: int


class Language(BaseModel):
    language: str
    proficiency: Literal["basic", "conversational", "professional", "native"]


class SalaryRange(BaseModel):
    """Expected salary in INR Lakhs Per Annum."""
    min: float = Field(ge=0)
    max: float = Field(ge=0)


class RedrobSignals(BaseModel):
    """Simulated platform activity and engagement signals from the Redrob ecosystem."""

    profile_completeness_score: float = Field(ge=0, le=100, description="Percentage of profile completeness.")
    signup_date: str
    last_active_date: str
    open_to_work_flag: bool
    profile_views_received_30d: int = Field(ge=0)
    applications_submitted_30d: int = Field(ge=0)
    recruiter_response_rate: float = Field(ge=0, le=1, description="Fraction of recruiter messages the candidate has responded to.")
    avg_response_time_hours: float = Field(ge=0)
    skill_assessment_scores: Dict[str, float] = Field(description="Dict of skill_name -> score 0-100. Assessments completed on Redrob platform.")
    connection_count: int = Field(ge=0)
    endorsements_received: int = Field(ge=0)
    notice_period_days: int = Field(ge=0, le=180)
    expected_salary_range_inr_lpa: SalaryRange = Field(description="Expected salary in INR Lakhs Per Annum.")
    preferred_work_mode: Literal["remote", "hybrid", "onsite", "flexible"]
    willing_to_relocate: bool
    github_activity_score: float = Field(ge=-1, le=100, description="0-100 score based on commits, PRs, stars in last 12 months. -1 if no GitHub linked.")
    search_appearance_30d: int = Field(ge=0, description="Number of times profile appeared in recruiter searches in last 30 days.")
    saved_by_recruiters_30d: int = Field(ge=0, description="Number of recruiters who saved this profile in last 30 days.")
    interview_completion_rate: float = Field(ge=0, le=1, description="Fraction of scheduled interviews actually attended.")
    offer_acceptance_rate: float = Field(ge=-1, le=1, description="Historical offer acceptance rate. -1 if no offer history.")
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool


class Candidate(BaseModel):
    """Schema for a single candidate profile in the Intelligent Candidate Discovery & Ranking Challenge dataset."""

    candidate_id: str = Field(
        pattern=r"^CAND_[0-9]{7}$",
        description="Unique identifier for the candidate. Format: CAND_XXXXXXX (7 digits).",
    )
    profile: Profile
    career_history: List[CareerHistory] = Field(
        min_length=1,
        max_length=10,
    )
    education: List[Education] = Field(default_factory=list, max_length=5)
    skills: List[Skill] = Field(default_factory=list)
    certifications: Optional[List[Certification]] = []
    languages: Optional[List[Language]] = []
    redrob_signals: RedrobSignals = Field(description="Simulated platform activity and engagement signals from the Redrob ecosystem.")
