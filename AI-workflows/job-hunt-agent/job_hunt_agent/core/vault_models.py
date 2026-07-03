"""Typed representations of vault-Resume content.

These models mirror the markdown conventions actually observed in the vault
(see vault_reader.py for the parsing logic that populates them). The vault is
edited outside of coding sessions, so every list here is populated
best-effort — a single malformed block should never prevent the rest of the
vault from loading (see VaultSnapshot.warnings).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

KNOWN_VARIANTS = ("data-science", "bioinformatics", "ai-engineering")

KNOWN_REGISTERS = (
    "formal-professional",
    "casual-direct",
    "teaching-mentoring",
    "stopgap-service",
)

DEFAULT_EXCLUDED_ASPIRATIONAL_SKILLS = ["CrewAI", "RAG", "Finetuning", "Hugging Face"]
DEFAULT_FORBIDDEN_TERMS = ["FDA"]


class ExperienceBullet(BaseModel):
    bullet_id: str
    employer: str
    used_in: list[str] = Field(default_factory=list)
    relevance_note: str | None = None
    other_notes: list[str] = Field(default_factory=list)
    text: str
    superseded_by: str | None = Field(
        default=None,
        description="bullet_id of the newer bullet this one was retired/merged into, "
        "parsed from a '**Used in:** none (retired — superseded by `X`)' line, if present.",
    )


class EmployerExperience(BaseModel):
    employer: str
    title: str
    dates: str
    intro_notes: str | None = None
    bullets: list[ExperienceBullet] = Field(default_factory=list)
    file_path: Path


class UsedInEntry(BaseModel):
    keywords: list[str]
    variants: list[str]
    raw_line: str


class SkillFile(BaseModel):
    title: str
    category_path: str
    used_entries: list[UsedInEntry] = Field(default_factory=list)
    all_keywords: list[str] = Field(default_factory=list)
    available_not_yet_used: list[str] = Field(default_factory=list)
    available_note_raw: str | None = None
    file_path: Path


class ResumeVariant(BaseModel):
    variant: str
    lane: str
    subtitle: str
    audience: str
    has_selected_projects: bool = False
    selected_projects: list[str] = Field(default_factory=list)
    education_order: list[str] = Field(default_factory=list)
    reviewed: bool = False
    last_reviewed: str | None = None
    converted_to_docx_or_pdf: bool = False
    used_for_applications: list[str] = Field(default_factory=list)
    summary_text: str = ""
    skills_section_raw: str = ""
    bullet_ids_used: list[str] = Field(default_factory=list)
    file_path: Path


class ProjectEntry(BaseModel):
    entry_id: str
    used_in: list[str] = Field(default_factory=list)
    title: str = ""
    dates: str = ""
    text: str


class PatentPublicationEntry(BaseModel):
    entry_id: str
    used_in: list[str] = Field(default_factory=list)
    text: str


class EducationEntry(BaseModel):
    entry_id: str
    text: str


class CoverLetterVoiceExample(BaseModel):
    company: str
    role: str
    letter_register: str
    full_text: str
    notes: str = ""
    file_path: Path


class AchievementParagraph(BaseModel):
    block_id: str
    source: str | None = None
    maps_to: str | None = None
    letter_register: str | None = None
    text: str


class BuildingBlock(BaseModel):
    block_id: str
    letter_register: str | None = None
    text: str
    needs_human_edit: bool = False
    is_fragment: bool = Field(
        default=False,
        description="True when block_id ends in '-fragment' — a phrase meant to be worked "
        "into other prose (e.g. the company-specific paragraph), not rendered as a "
        "standalone sentence on its own.",
    )
    extra_fields: dict[str, str] = Field(default_factory=dict)


class VaultSnapshot(BaseModel):
    loaded_at: datetime
    vault_path: Path
    variants: dict[str, ResumeVariant] = Field(default_factory=dict)
    experience: dict[str, EmployerExperience] = Field(default_factory=dict)
    skills: list[SkillFile] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)
    patents: list[PatentPublicationEntry] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    voice_examples: list[CoverLetterVoiceExample] = Field(default_factory=list)
    achievement_paragraphs: list[AchievementParagraph] = Field(default_factory=list)
    soft_skills: list[BuildingBlock] = Field(default_factory=list)
    greetings: list[BuildingBlock] = Field(default_factory=list)
    closings: list[BuildingBlock] = Field(default_factory=list)
    excluded_aspirational_skills: list[str] = Field(
        default_factory=lambda: list(DEFAULT_EXCLUDED_ASPIRATIONAL_SKILLS)
    )
    forbidden_terms: list[str] = Field(default_factory=lambda: list(DEFAULT_FORBIDDEN_TERMS))
    warnings: list[str] = Field(default_factory=list)

    def all_experience_bullets(self) -> list[ExperienceBullet]:
        bullets: list[ExperienceBullet] = []
        for exp in self.experience.values():
            bullets.extend(exp.bullets)
        return bullets
