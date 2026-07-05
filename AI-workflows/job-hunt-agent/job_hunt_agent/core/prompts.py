"""Prompt builders for the single job-matching LLM call.

Only one call is made per posting (see matcher.py's two-phase design): a
deterministic keyword prefilter narrows the vault down to candidate skills
and bullets first, then this module builds one prompt asking the model to
reason comparatively over all three resume variants at once.
"""

from __future__ import annotations

from job_hunt_agent.core.models import JobPosting
from job_hunt_agent.core.vault_models import ExperienceBullet, KNOWN_VARIANTS, SkillFile, VaultSnapshot

SYSTEM_MATCHER = """You are helping match a real job posting against a candidate's \
existing, evidenced career-history vault (resumes, experience bullets, and skills). \
You are NOT writing new claims — every skill or bullet you surface must come from \
the vault content given to you verbatim. Do not invent achievements, skills, or \
metrics that are not present in the provided vault content.

Score all three resume variants comparatively in a single pass, not independently \
— your fit_score values should be meaningful relative to each other for this one \
posting. Read each candidate bullet's relevance_note/other_notes carefully before \
scoring: the vault's own prior human judgment about why a bullet suits one \
audience over another is directly relevant to your reasoning.

Pay special attention to "available, not yet used" skills content — surfacing real, \
evidenced-but-unused vault content that fits this specific posting is one of the \
most valuable things you can do here, more valuable than only re-confirming content \
already in the recommended variant.

Hard constraints, never violate these:
{constraints}
"""


def _format_constraints(vault: VaultSnapshot) -> str:
    lines = []
    for term in vault.forbidden_terms:
        lines.append(f"- Never mention or imply \"{term}\" in any reasoning or surfaced content.")
    for skill in vault.excluded_aspirational_skills:
        lines.append(
            f"- Never surface or recommend \"{skill}\" as a skill — it is aspirational, "
            "not yet a real, evidenced skill."
        )
    return "\n".join(lines)


def _format_variant_summaries(vault: VaultSnapshot) -> str:
    blocks = []
    for slug in KNOWN_VARIANTS:
        variant = vault.variants.get(slug)
        if variant is None:
            continue
        blocks.append(
            f"### {slug}\n"
            f"Lane: {variant.lane}\n"
            f"Subtitle: {variant.subtitle}\n"
            f"Audience: {variant.audience}\n"
            f"Summary: {variant.summary_text}"
        )
    return "\n\n".join(blocks)


def _format_candidate_bullets(bullets: list[ExperienceBullet]) -> str:
    if not bullets:
        return "(no candidate bullets surfaced by the prefilter)"
    blocks = []
    for b in bullets:
        note_parts = []
        if b.relevance_note:
            note_parts.append(f"Relevance: {b.relevance_note}")
        note_parts.extend(b.other_notes)
        notes = " | ".join(note_parts) if note_parts else "(no notes)"
        blocks.append(
            f"- id: {b.bullet_id} | employer: {b.employer} | used_in: {', '.join(b.used_in) or '(none)'}\n"
            f"  text: {b.text}\n"
            f"  notes: {notes}"
        )
    return "\n".join(blocks)


def _format_candidate_skills(skills: list[SkillFile]) -> str:
    if not skills:
        return "(no candidate skill files surfaced by the prefilter)"
    blocks = []
    for s in skills:
        used_summary = "; ".join(
            f"{', '.join(e.keywords)} -> {', '.join(e.variants)}" for e in s.used_entries
        ) or "(nothing currently used)"
        available = ", ".join(s.available_not_yet_used[:20]) or "(none)"
        blocks.append(
            f"- {s.title} [{s.category_path}]\n"
            f"  currently used: {used_summary}\n"
            f"  available but not yet used: {available}"
        )
    return "\n".join(blocks)


def _format_registers(vault: VaultSnapshot) -> str:
    if not vault.greetings:
        return "(no register examples available)"
    registers = sorted({g.block_id for g in vault.greetings})
    return ", ".join(registers)


def _format_achievement_paragraphs(vault: VaultSnapshot) -> str:
    if not vault.achievement_paragraphs:
        return "(none available)"
    return "\n".join(
        f"- id: {p.block_id} | register: {p.letter_register or '(unspecified)'} | text: {p.text}"
        for p in vault.achievement_paragraphs
    )


def _format_soft_skills(vault: VaultSnapshot) -> str:
    if not vault.soft_skills:
        return "(none available)"
    return "\n".join(
        f"- id: {s.block_id} | register: {s.letter_register or '(unspecified)'}"
        f"{' | NEEDS HUMAN EDIT, avoid unless nothing else fits' if s.needs_human_edit else ''}"
        f" | text: {s.text}"
        for s in vault.soft_skills
    )


def build_match_prompt(
    job: JobPosting,
    vault: VaultSnapshot,
    candidate_skills: list[SkillFile],
    candidate_bullets: list[ExperienceBullet],
) -> str:
    """Build the single user-turn prompt for the matching LLM call."""
    return f"""## Job posting

Company: {job.company or "(unknown)"}
Role: {job.role_title or "(unknown)"}

{job.raw_text}

## The three resume variants (score all three)

{_format_variant_summaries(vault)}

## Candidate experience bullets (from the deterministic keyword prefilter)

{_format_candidate_bullets(candidate_bullets)}

## Candidate skill files (from the deterministic keyword prefilter)

{_format_candidate_skills(candidate_skills)}

## Available cover-letter registers

{_format_registers(vault)}

## Available achievement paragraphs (for step 5 below — pick only from these real IDs)

{_format_achievement_paragraphs(vault)}

## Available soft-skill blocks (for step 5 below — pick only from these real IDs)

{_format_soft_skills(vault)}

## Your task

1. Score all three variants (data-science, bioinformatics, ai-engineering) for fit \
against this specific posting, with reasoning grounded in the candidate bullets/skills \
above.
2. Recommend exactly one variant as the best starting point.
3. Recommend a cover-letter register from the list above, with reasoning based on the \
posting's tone/company context.
4. Surface any candidate skills or bullets above (marked "available but not yet used" \
where relevant) that are a strong fit for this posting but aren't in any resume variant \
yet — this is the most valuable output of this analysis.
5. Recommend which achievement-paragraph ID(s) and which soft-skill ID would fit a cover \
letter for this posting, choosing only from real IDs implied by the vault content above.
"""


def build_system_prompt(vault: VaultSnapshot) -> str:
    return SYSTEM_MATCHER.format(constraints=_format_constraints(vault))
