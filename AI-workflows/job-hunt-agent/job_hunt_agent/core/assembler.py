"""Deterministic draft resume/cover-letter assembly — no LLM calls.

All judgment (variant choice, register, achievement-paragraph/soft-skill picks)
already happened in the single matching LLM call (see matcher.py). Assembly
here is pure: resolve IDs into text and render Jinja2 templates. Every writer
in this module only ever writes under the caller-supplied output_dir — never
into vault_path, which is treated as strictly read-only input.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from job_hunt_agent.core.guardrails import scan_for_violations
from job_hunt_agent.core.models import DraftCoverLetter, DraftResume, JobMatchResult
from job_hunt_agent.core.vault_models import VaultSnapshot

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATES_DIR)),
    autoescape=select_autoescape(enabled_extensions=()),
    trim_blocks=True,
    lstrip_blocks=True,
)


def slugify(*parts: str) -> str:
    joined = "-".join(p for p in parts if p)
    slug = re.sub(r"[^a-z0-9]+", "-", joined.lower()).strip("-")
    return slug or "untitled"


def _first_template_option(text: str) -> str:
    """Greetings.md sometimes lists alternates as `Option A` / `Option B` —
    pick the first and strip the backtick-wrapped markdown formatting, since
    this is going into a plain-text draft, not being displayed in Obsidian.
    """
    if not text:
        return text
    first = text.split(" / ")[0].strip()
    if len(first) >= 2 and first.startswith("`") and first.endswith("`"):
        first = first[1:-1]
    return first


def assemble_draft_resume(
    match: JobMatchResult,
    vault: VaultSnapshot,
    output_dir: Path,
    variant_override: str | None = None,
    include_surfaced: bool = True,
) -> DraftResume:
    if match.llm_output is None:
        raise ValueError("cannot assemble a draft resume from a JobMatchResult with no llm_output")

    variant_slug = variant_override or match.llm_output.recommended_variant
    variant = vault.variants.get(variant_slug)
    if variant is None:
        raise ValueError(f"unknown resume variant: {variant_slug!r}")

    employer_bullets = []
    for exp in vault.experience.values():
        bullets = [b for b in exp.bullets if variant_slug in b.used_in]
        if bullets:
            employer_bullets.append({"employer": exp, "bullets": bullets})

    surfaced_bullets = []
    surfaced_skills = []
    if include_surfaced:
        bullet_lookup = {b.bullet_id: b for b in vault.all_experience_bullets()}
        for sb in match.llm_output.surfaced_bullets:
            full = bullet_lookup.get(sb.bullet_id)
            if full is None:
                continue
            if variant_slug in full.used_in:
                # Ground-truth check, not LLM-trust: the vault's own `used_in`
                # tag says this bullet is already in the recommended variant's
                # Experience section, despite the LLM having called it
                # "surfaced" (implying unused). Silently drop rather than
                # render a bullet twice — the prompt asks the model not to do
                # this, but compliance isn't guaranteed, so this is enforced
                # deterministically here instead of trusted from the LLM output.
                continue
            surfaced_bullets.append({"bullet": full, "why_relevant": sb.why_relevant})

        already_used_keywords = {
            kw.lower()
            for skill_file in vault.skills
            for entry in skill_file.used_entries
            if variant_slug in entry.variants
            for kw in entry.keywords
        }
        surfaced_skills = [
            s
            for s in match.llm_output.surfaced_skills
            if s.keyword.lower() not in already_used_keywords
        ]

    projects = (
        [p for p in vault.projects if variant_slug in p.used_in]
        if variant.has_selected_projects
        else []
    )

    education_by_id = {e.entry_id: e for e in vault.education}
    ordered_education = [
        education_by_id[eid] for eid in variant.education_order if eid in education_by_id
    ] or vault.education

    template = _env.get_template("resume_draft.md.j2")
    rendered = template.render(
        variant=variant,
        employer_bullets=employer_bullets,
        surfaced_bullets=surfaced_bullets,
        surfaced_skills=surfaced_skills,
        patents=vault.patents,
        education=ordered_education,
        projects=projects,
        company=match.job.company,
        role=match.job.role_title,
        url=match.job.url,
        generated_at=datetime.now().isoformat(timespec="seconds"),
    )

    warnings = scan_for_violations(rendered, vault)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "resume.md"
    out_path.write_text(rendered, encoding="utf-8")

    return DraftResume(variant=variant_slug, output_path=out_path, warnings=warnings)


def assemble_draft_cover_letter(
    match: JobMatchResult,
    vault: VaultSnapshot,
    output_dir: Path,
    register_override: str | None = None,
) -> DraftCoverLetter:
    if match.llm_output is None:
        raise ValueError(
            "cannot assemble a draft cover letter from a JobMatchResult with no llm_output"
        )

    register = register_override or match.llm_output.cover_letter_register

    greeting = next((g for g in vault.greetings if g.letter_register == register), None)
    if greeting is None and vault.greetings:
        greeting = vault.greetings[0]
    salutation = _first_template_option(
        greeting.extra_fields.get("salutation", "") if greeting else "Dear Hiring Manager,"
    )
    opening_line = _first_template_option(
        greeting.extra_fields.get("opening line", "") if greeting else ""
    )
    opening_line = opening_line.replace("[ROLE]", match.job.role_title or "[ROLE]").replace(
        "[COMPANY]", match.job.company or "[COMPANY]"
    )

    closing = next((c for c in vault.closings if c.letter_register == register), None)
    if closing is None and vault.closings:
        closing = vault.closings[0]
    closing_text = closing.text if closing else "Sincerely,"

    achievement_paragraphs = [
        p
        for p in vault.achievement_paragraphs
        if p.block_id in match.llm_output.recommended_achievement_paragraph_ids
    ]

    soft_skill = next(
        (s for s in vault.soft_skills if s.block_id == match.llm_output.recommended_soft_skill_id),
        None,
    )

    template = _env.get_template("cover_letter_draft.md.j2")
    rendered = template.render(
        company=match.job.company,
        role=match.job.role_title,
        url=match.job.url,
        register=register,
        salutation=salutation,
        opening_line=opening_line,
        achievement_paragraphs=achievement_paragraphs,
        soft_skill=soft_skill,
        closing_text=closing_text,
        generated_at=datetime.now().isoformat(timespec="seconds"),
    )

    warnings = scan_for_violations(rendered, vault)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "cover_letter.md"
    out_path.write_text(rendered, encoding="utf-8")

    return DraftCoverLetter(
        company=match.job.company or "unknown",
        role=match.job.role_title or "unknown",
        letter_register=register,
        output_path=out_path,
        warnings=warnings,
    )
