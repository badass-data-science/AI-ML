"""Pipeline-facing Pydantic models: company config, a raw fetched posting, its
deterministic ghost-risk score, and the seen-store record that makes staleness
meaningful across multiple `pull` runs.

Mirrors job-hunt-agent's core/models.py pattern (a raw input, a scored/derived
output, and a combined result), but with a deterministic score instead of an
LLM-produced one — job-radar makes no LLM calls at all.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

ATSPlatform = Literal["greenhouse", "lever", "ashby"]


class CompanyConfig(BaseModel):
    name: str
    ats: ATSPlatform
    slug: str = Field(description="The company's board token / site id on that ATS platform.")


class RawPosting(BaseModel):
    external_id: str = Field(description="The posting's id as assigned by the source ATS.")
    company: str
    ats: ATSPlatform
    title: str
    location: str | None = None
    department: str | None = None
    url: str
    raw_text: str = Field(description="Plain-text/markdown job description, HTML already stripped.")
    posted_at: datetime | None = None
    updated_at: datetime | None = None


class GhostSignal(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="Higher = more likely a ghost/stale posting.")
    reasons: list[str] = Field(
        default_factory=list, description="Every signal that contributed to the score, human-readable."
    )


class SeenRecord(BaseModel):
    key: str = Field(description="f'{ats}:{slug}:{external_id}' — unique across all sources.")
    first_seen_at: datetime
    last_seen_at: datetime
    seen_count: int = 1
    last_title: str
    last_source_updated_at: datetime | None = None


class ScoredPosting(BaseModel):
    posting: RawPosting
    ghost: GhostSignal
    first_seen_at: datetime
    last_seen_at: datetime
    seen_count: int


class DiscoveredSlug(BaseModel):
    ats: ATSPlatform
    slug: str
    job_count: int = Field(description="Number of live postings found at this slug — confirms it's a real, active board.")
