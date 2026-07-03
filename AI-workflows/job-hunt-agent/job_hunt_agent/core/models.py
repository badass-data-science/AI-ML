"""Pipeline-facing Pydantic models: job postings, matching output, drafts, and
the application tracker's record type.

Mirrors strategic-reports' RawArticle -> ArticleSummary/StrategicInsight ->
TopicResult pattern: a raw input, an LLM-produced structured output, and a
final error-isolating container that's the only thing crossing into the
CLI/rendering layer.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from job_hunt_agent.core.vault_models import KNOWN_REGISTERS, KNOWN_VARIANTS

APPLICATION_STATUSES = (
    "drafted",
    "applied",
    "phone_screen",
    "interview",
    "offer",
    "rejected",
    "withdrawn",
)


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class JobPosting(BaseModel):
    raw_text: str
    source_path: Path | None = None
    company: str | None = None
    role_title: str | None = None
    url: str | None = Field(default=None, description="The original posting's apply/listing URL, if known.")
    ghost_score: float | None = Field(
        default=None, description="job-radar's ghost-risk score for this posting (0-1), if known."
    )
    ghost_reasons: list[str] = Field(
        default_factory=list, description="job-radar's ghost-risk reasons for this posting, if known."
    )
    fetched_at: datetime = Field(default_factory=datetime.now)


class VariantScore(BaseModel):
    variant: Literal["data-science", "bioinformatics", "ai-engineering"]
    fit_score: float = Field(ge=0.0, le=1.0)
    reasoning: list[str] = Field(min_length=2, max_length=4)
    strengths: list[str] = Field(min_length=1, max_length=5)
    gaps: list[str] = Field(default_factory=list, max_length=5)


class SurfacedSkill(BaseModel):
    file_title: str = Field(
        description="The exact Skills file title this keyword came from, e.g. 'Database Skills' "
        "— copy it verbatim from the candidate skill files list, do not invent one."
    )
    keyword: str = Field(
        description="The exact keyword or phrase from that file's 'available but not yet used' "
        "list — copy it verbatim, do not paraphrase or invent a new skill name."
    )
    why_relevant: str = Field(
        description="One sentence on why this specific unused keyword fits this specific posting."
    )


class SurfacedBullet(BaseModel):
    bullet_id: str = Field(
        description="The exact bullet id from the candidate experience bullets list above, e.g. "
        "'tfs-neo4j-graph-database' — copy it verbatim, do not invent one."
    )
    employer: str = Field(description="The employer that bullet came from, copied from the candidate list.")
    why_relevant: str = Field(
        description="One sentence on why this specific unused bullet fits this specific posting."
    )


class JobMatchLLMOutput(BaseModel):
    """The instructor `response_model` for the single matching LLM call."""

    variant_scores: list[VariantScore] = Field(min_length=3, max_length=3)
    recommended_variant: str
    cover_letter_register: str
    register_reasoning: str
    surfaced_skills: list[SurfacedSkill] = Field(default_factory=list, max_length=8)
    surfaced_bullets: list[SurfacedBullet] = Field(default_factory=list, max_length=5)
    recommended_achievement_paragraph_ids: list[str] = Field(min_length=1, max_length=2)
    recommended_soft_skill_id: str

    @field_validator("recommended_variant")
    @classmethod
    def _valid_variant(cls, v: str) -> str:
        if v not in KNOWN_VARIANTS:
            raise ValueError(
                f"recommended_variant must be one of {KNOWN_VARIANTS}, got {v!r}"
            )
        return v

    @field_validator("cover_letter_register")
    @classmethod
    def _valid_register(cls, v: str) -> str:
        if v not in KNOWN_REGISTERS:
            raise ValueError(f"cover_letter_register must be one of {KNOWN_REGISTERS}, got {v!r}")
        return v


class JobMatchResult(BaseModel):
    job: JobPosting
    llm_output: JobMatchLLMOutput | None = None
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    error: str | None = None


class DraftResume(BaseModel):
    variant: str
    output_path: Path
    warnings: list[str] = Field(default_factory=list)


class DraftCoverLetter(BaseModel):
    company: str
    role: str
    letter_register: str
    output_path: Path
    warnings: list[str] = Field(default_factory=list)


class ApplicationRecord(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    company: str
    role: str
    date_applied: date | None = None
    resume_variant: str
    cover_letter_register: str
    status: Literal[
        "drafted", "applied", "phone_screen", "interview", "offer", "rejected", "withdrawn"
    ] = "drafted"
    outcome: str | None = None
    notes: str | None = None
    job_posting_path: str | None = None
    draft_resume_path: str | None = None
    draft_cover_letter_path: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
