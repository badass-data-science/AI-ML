"""Parse vault-Resume markdown into typed VaultSnapshot objects.

The vault is edited outside of coding sessions (by Emily, directly in
Obsidian), so parsing here is deliberately lenient: a single malformed block
or unexpected line is recorded in VaultSnapshot.warnings and skipped, never
raised. One bad file must never prevent the other 30+ files from loading.

vault-Resume is treated as strictly read-only input by this module and by
every caller of it — nothing in this package ever writes into vault_path.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import frontmatter

from job_hunt_agent.core.vault_models import (
    DEFAULT_EXCLUDED_ASPIRATIONAL_SKILLS,
    DEFAULT_FORBIDDEN_TERMS,
    KNOWN_REGISTERS,
    KNOWN_VARIANTS,
    AchievementParagraph,
    BuildingBlock,
    CoverLetterVoiceExample,
    EducationEntry,
    EmployerExperience,
    ExperienceBullet,
    PatentPublicationEntry,
    ProjectEntry,
    ResumeVariant,
    SkillFile,
    UsedInEntry,
    VaultSnapshot,
)

_BULLET_HEADING_RE = re.compile(r"^### (.+)$", re.MULTILINE)
_H2_HEADING_RE = re.compile(r"^## (.+)$", re.MULTILINE)
_BOLD_LABEL_RE = re.compile(r"^\*\*([A-Za-z ]+?):\*\*\s*(.*)$")
_BULLET_COMMENT_RE = re.compile(r"<!--\s*bullets:\s*(.*?)\s*-->")
_SUPERSEDED_BY_RE = re.compile(r"superseded by `([\w-]+)`")
_WIKILINK_HEADING_RE = re.compile(r"\[\[([^\]#|]+)#([^\]|]+)(?:\|[^\]]+)?\]\]")


def load_vault(vault_path: Path) -> VaultSnapshot:
    """Single public entry point: parse the whole vault into a VaultSnapshot."""
    warnings: list[str] = []
    vault_path = Path(vault_path)

    variants = _load_variants(vault_path, warnings)
    experience = _load_experience(vault_path, warnings)
    skills = _load_skills(vault_path, warnings)
    projects = _load_id_tagged_entries(
        vault_path / "Resumes" / "projects.md", ProjectEntry, warnings
    )
    patents = _load_id_tagged_entries(
        vault_path / "Resumes" / "patents-and-publications.md",
        PatentPublicationEntry,
        warnings,
    )
    education = _load_education(vault_path / "Resumes" / "education.md", warnings)
    voice_examples = _load_voice_examples(vault_path, warnings)
    achievement_paragraphs = _load_achievement_paragraphs(vault_path, warnings)
    soft_skills = _load_building_blocks(
        vault_path / "CoverLetters" / "building-blocks" / "soft-skills-and-work-ethic.md",
        warnings,
    )
    greetings = _load_building_blocks(
        vault_path / "CoverLetters" / "building-blocks" / "greetings.md", warnings
    )
    closings = _load_building_blocks(
        vault_path / "CoverLetters" / "building-blocks" / "closings.md", warnings
    )
    excluded_aspirational_skills, forbidden_terms = _load_exclusion_rules(
        vault_path, warnings
    )

    return VaultSnapshot(
        loaded_at=datetime.now(),
        vault_path=vault_path,
        variants=variants,
        experience=experience,
        skills=skills,
        projects=projects,
        patents=patents,
        education=education,
        voice_examples=voice_examples,
        achievement_paragraphs=achievement_paragraphs,
        soft_skills=soft_skills,
        greetings=greetings,
        closings=closings,
        excluded_aspirational_skills=excluded_aspirational_skills,
        forbidden_terms=forbidden_terms,
        warnings=warnings,
    )


def _extract_section(body: str, heading: str) -> str:
    """Return the text of a '## {heading}' section, up to the next '## ' heading."""
    pattern = re.compile(
        rf"^## {re.escape(heading)}\s*\n(.*?)(?=^## |\Z)", re.MULTILINE | re.DOTALL
    )
    match = pattern.search(body)
    return match.group(1).strip() if match else ""


# ---------------------------------------------------------------------------
# Resumes/variants/*.md
# ---------------------------------------------------------------------------


def _load_variants(vault_path: Path, warnings: list[str]) -> dict[str, ResumeVariant]:
    variants: dict[str, ResumeVariant] = {}
    variants_dir = vault_path / "Resumes" / "variants"
    if not variants_dir.is_dir():
        warnings.append(f"variants directory not found: {variants_dir}")
        return variants

    for path in sorted(variants_dir.glob("*.md")):
        try:
            post = frontmatter.load(path)
            body = post.content
            meta = post.metadata
            variant_slug = meta.get("variant") or path.stem
            summary_text = _extract_section(body, "Professional Summary")
            skills_section_raw = _extract_section(body, "Skills")
            bullet_ids_used = []
            for comment in _BULLET_COMMENT_RE.findall(body):
                for token in comment.split(","):
                    token = re.sub(r"\(.*?\)", "", token).strip()
                    if token:
                        bullet_ids_used.append(token)

            variants[variant_slug] = ResumeVariant(
                variant=variant_slug,
                lane=meta.get("lane", ""),
                subtitle=meta.get("subtitle", ""),
                audience=meta.get("audience", ""),
                has_selected_projects=bool(meta.get("has_selected_projects", False)),
                selected_projects=list(meta.get("selected_projects", []) or []),
                education_order=list(meta.get("education_order", []) or []),
                reviewed=bool(meta.get("reviewed", False)),
                last_reviewed=meta.get("last_reviewed"),
                converted_to_docx_or_pdf=bool(meta.get("converted_to_docx_or_pdf", False)),
                used_for_applications=list(meta.get("used_for_applications", []) or []),
                summary_text=summary_text,
                skills_section_raw=skills_section_raw,
                bullet_ids_used=bullet_ids_used,
                file_path=path,
            )
        except Exception as exc:  # noqa: BLE001 - deliberate error isolation
            warnings.append(f"failed to parse variant file {path}: {exc}")

    return variants


# ---------------------------------------------------------------------------
# Resumes/experience/*.md
# ---------------------------------------------------------------------------


def _load_experience(
    vault_path: Path, warnings: list[str]
) -> dict[str, EmployerExperience]:
    experience: dict[str, EmployerExperience] = {}
    experience_dir = vault_path / "Resumes" / "experience"
    if not experience_dir.is_dir():
        warnings.append(f"experience directory not found: {experience_dir}")
        return experience

    for path in sorted(experience_dir.glob("*.md")):
        try:
            exp = _parse_experience_file(path, warnings)
            experience[path.stem] = exp
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"failed to parse experience file {path}: {exc}")

    return experience


def _parse_experience_file(path: Path, warnings: list[str]) -> EmployerExperience:
    post = frontmatter.load(path)
    body = post.content
    meta = post.metadata

    bullets_section = _extract_section(body, "Bullets")
    intro_match = re.search(
        r"^## Bullets\s*\n", body, re.MULTILINE
    )
    intro_notes = None
    if intro_match:
        before_bullets = body[: intro_match.start()]
        # strip the H1 title + date line, keep any free prose after it
        lines = before_bullets.splitlines()
        prose_lines = [
            ln for ln in lines if ln.strip() and not ln.startswith("#") and not ln.startswith("*")
        ]
        if prose_lines:
            intro_notes = "\n".join(prose_lines).strip()

    bullets: list[ExperienceBullet] = []
    headings = list(_BULLET_HEADING_RE.finditer(bullets_section))
    for i, heading_match in enumerate(headings):
        bullet_id = heading_match.group(1).strip()
        start = heading_match.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(bullets_section)
        block = bullets_section[start:end].strip()

        used_in: list[str] = []
        relevance_note: str | None = None
        other_notes: list[str] = []
        text_lines: list[str] = []
        superseded_by: str | None = None

        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            label_match = _BOLD_LABEL_RE.match(line)
            if label_match:
                label, value = label_match.group(1).strip(), label_match.group(2).strip()
                if label.lower() == "used in":
                    used_in = [v.strip() for v in value.split(",") if v.strip()]
                    # A retired/merged bullet's "Used in" line reads like
                    # "none (retired — superseded by `bbh-topic-modeling`)" —
                    # doesn't change used_in's existing value (still whatever
                    # raw string it parses to today, unchanged behavior for
                    # any other reader of it), just additionally captures the
                    # replacement bullet_id so assembler.py's dedup check can
                    # ground-truth against it later.
                    superseded_match = _SUPERSEDED_BY_RE.search(value)
                    if superseded_match:
                        superseded_by = superseded_match.group(1)
                elif label.lower() == "relevance":
                    relevance_note = value
                else:
                    other_notes.append(f"{label}: {value}")
            else:
                text_lines.append(line)

        if not used_in:
            warnings.append(
                f"{path.name}: bullet '{bullet_id}' missing '**Used in:**' line"
            )

        bullets.append(
            ExperienceBullet(
                bullet_id=bullet_id,
                employer=meta.get("employer", path.stem),
                used_in=used_in,
                relevance_note=relevance_note,
                other_notes=other_notes,
                text=" ".join(text_lines).strip(),
                superseded_by=superseded_by,
            )
        )

    return EmployerExperience(
        employer=meta.get("employer", path.stem),
        title=meta.get("title", ""),
        dates=meta.get("dates", ""),
        intro_notes=intro_notes,
        bullets=bullets,
        file_path=path,
    )


# ---------------------------------------------------------------------------
# Skills/**/*.md
# ---------------------------------------------------------------------------


def _load_skills(vault_path: Path, warnings: list[str]) -> list[SkillFile]:
    skills: list[SkillFile] = []
    skills_dir = vault_path / "Skills"
    if not skills_dir.is_dir():
        warnings.append(f"Skills directory not found: {skills_dir}")
        return skills

    for path in sorted(skills_dir.rglob("*.md")):
        if path.name == "README.md":
            continue
        try:
            skills.append(_parse_skill_file(skills_dir, path, warnings))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"failed to parse skill file {path}: {exc}")

    return skills


def _parse_skill_file(skills_dir: Path, path: Path, warnings: list[str]) -> SkillFile:
    text = path.read_text(encoding="utf-8")
    title_match = re.search(r"^# (.+)$", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else path.stem

    used_section = _extract_section(text, "Used in current resumes")
    used_entries: list[UsedInEntry] = []
    available_note_raw: str | None = None

    for raw_line in used_section.splitlines():
        line = raw_line.strip().lstrip("*").strip()
        if not line:
            continue
        for clause in line.split(";"):
            clause = clause.strip()
            if not clause:
                continue
            if "→" in clause:
                left, right = clause.split("→", 1)
                keywords = [k.strip() for k in left.split(",") if k.strip()]
                variants_found = [v for v in KNOWN_VARIANTS if v in right]
                used_entries.append(
                    UsedInEntry(keywords=keywords, variants=variants_found, raw_line=clause)
                )
            else:
                # Any non-"→" line under this heading is a contextual/availability note
                # (e.g. "nothing in this file is used yet" or a cross-reference aside) —
                # kept verbatim rather than discarded, since it's exactly the kind of
                # unused-content signal the matcher needs, even when wording varies.
                available_note_raw = (
                    clause if available_note_raw is None else f"{available_note_raw} {clause}"
                )

    all_keywords: list[str] = []
    for h2_match in _H2_HEADING_RE.finditer(text):
        heading = h2_match.group(1).strip()
        if heading == "Used in current resumes":
            continue
        section_text = _extract_section(text, heading)
        for line in section_text.splitlines():
            line = line.strip()
            if line.startswith("* "):
                all_keywords.append(line[2:].strip())

    used_text_lower = used_section.lower()
    available_not_yet_used = [
        kw for kw in all_keywords if kw.lower() not in used_text_lower
    ]

    category_path = str(path.relative_to(skills_dir).parent)

    return SkillFile(
        title=title,
        category_path=category_path,
        used_entries=used_entries,
        all_keywords=all_keywords,
        available_not_yet_used=available_not_yet_used,
        available_note_raw=available_note_raw,
        file_path=path,
    )


# ---------------------------------------------------------------------------
# Shared "## {id}" + "**Used in:**" block parser — projects.md / patents.md
# ---------------------------------------------------------------------------


def _load_id_tagged_entries(path: Path, model_cls, warnings: list[str]) -> list:
    entries: list = []
    if not path.is_file():
        warnings.append(f"file not found: {path}")
        return entries

    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"failed to read {path}: {exc}")
        return entries

    accepts_title = "title" in model_cls.model_fields
    accepts_dates = "dates" in model_cls.model_fields

    headings = list(_H2_HEADING_RE.finditer(text))
    for i, heading_match in enumerate(headings):
        entry_id = heading_match.group(1).strip()
        if entry_id.lower() == "status":
            continue
        start = heading_match.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        block = text[start:end].strip()

        used_in: list[str] = []
        title = ""
        dates = ""
        bullet_lines: list[str] = []
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            label_match = _BOLD_LABEL_RE.match(line)
            if label_match:
                label = label_match.group(1).strip().lower()
                value = label_match.group(2).strip()
                if label == "used in":
                    used_in = [v.strip() for v in value.split(",") if v.strip()]
                elif label == "title":
                    title = value
                elif label == "dates":
                    dates = value
                # Any other label (Applications, Authors, Journal, Employer context,
                # Source, ...) is internal provenance/reasoning, not resume-facing
                # content — deliberately discarded here, not carried into drafts.
            elif line.startswith("- "):
                bullet_lines.append(line[2:].strip())
            # Free-form trailing commentary paragraphs (not a "- " bullet, not a
            # "**Label:**" line) are also internal reasoning — discarded.

        if not used_in:
            warnings.append(f"{path.name}: entry '{entry_id}' missing '**Used in:**' line")
        if not bullet_lines:
            warnings.append(
                f"{path.name}: entry '{entry_id}' has no '- ' bulleted resume-facing text"
            )

        kwargs = dict(entry_id=entry_id, used_in=used_in, text="\n".join(bullet_lines))
        if accepts_title:
            kwargs["title"] = title
        if accepts_dates:
            kwargs["dates"] = dates
        entries.append(model_cls(**kwargs))

    return entries


def _load_education(path: Path, warnings: list[str]) -> list[EducationEntry]:
    entries: list[EducationEntry] = []
    if not path.is_file():
        warnings.append(f"file not found: {path}")
        return entries

    text = path.read_text(encoding="utf-8")
    entries_section = _extract_section(text, "Entries")
    headings = list(_BULLET_HEADING_RE.finditer(entries_section))
    for i, heading_match in enumerate(headings):
        entry_id = heading_match.group(1).strip()
        start = heading_match.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(entries_section)
        block = entries_section[start:end].strip()
        entries.append(EducationEntry(entry_id=entry_id, text=block))

    return entries


# ---------------------------------------------------------------------------
# CoverLetters/
# ---------------------------------------------------------------------------


def _load_voice_examples(
    vault_path: Path, warnings: list[str]
) -> list[CoverLetterVoiceExample]:
    examples: list[CoverLetterVoiceExample] = []
    voice_dir = vault_path / "CoverLetters" / "voice-examples"
    if not voice_dir.is_dir():
        warnings.append(f"voice-examples directory not found: {voice_dir}")
        return examples

    for path in sorted(voice_dir.glob("*.md")):
        try:
            post = frontmatter.load(path)
            meta = post.metadata
            body = post.content
            full_text = _extract_section(body, "Full text")
            notes = _extract_section(body, "Notes")
            examples.append(
                CoverLetterVoiceExample(
                    company=meta.get("company", path.stem),
                    role=meta.get("role", ""),
                    letter_register=meta.get("register", ""),
                    full_text=full_text,
                    notes=notes,
                    file_path=path,
                )
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"failed to parse voice example {path}: {exc}")

    return examples


def _parse_metadata_block(block: str) -> tuple[dict[str, str], list[str]]:
    """Split a '## {id}' block into {label: value} metadata lines and remaining text lines."""
    meta: dict[str, str] = {}
    text_lines: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        label_match = _BOLD_LABEL_RE.match(stripped)
        if label_match:
            meta[label_match.group(1).strip().lower()] = label_match.group(2).strip()
        else:
            text_lines.append(stripped)
    return meta, text_lines


def _load_achievement_paragraphs(
    vault_path: Path, warnings: list[str]
) -> list[AchievementParagraph]:
    paragraphs: list[AchievementParagraph] = []
    path = vault_path / "CoverLetters" / "building-blocks" / "achievement-paragraphs.md"
    if not path.is_file():
        warnings.append(f"file not found: {path}")
        return paragraphs

    text = path.read_text(encoding="utf-8")
    headings = list(_H2_HEADING_RE.finditer(text))
    for i, heading_match in enumerate(headings):
        block_id = heading_match.group(1).strip()
        start = heading_match.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        block = text[start:end].strip()
        meta, text_lines = _parse_metadata_block(block)
        # quoted prose ("> ...") is the paragraph itself
        quoted = [ln.lstrip(">").strip() for ln in text_lines if ln.startswith(">")]
        prose = "\n".join(quoted) if quoted else " ".join(text_lines)

        paragraphs.append(
            AchievementParagraph(
                block_id=block_id,
                source=meta.get("source"),
                maps_to=meta.get("maps to"),
                letter_register=meta.get("register"),
                text=prose,
            )
        )

    return paragraphs


_NON_CONTENT_HEADINGS = {
    "when to use which",
    "signature variants observed",
    "usage guidance",
}


def _load_building_blocks(path: Path, warnings: list[str]) -> list[BuildingBlock]:
    blocks: list[BuildingBlock] = []
    if not path.is_file():
        warnings.append(f"file not found: {path}")
        return blocks

    text = path.read_text(encoding="utf-8")
    headings = list(_H2_HEADING_RE.finditer(text))
    for i, heading_match in enumerate(headings):
        block_id = heading_match.group(1).strip()
        if block_id.lower().startswith("when to") or block_id.lower() in _NON_CONTENT_HEADINGS:
            continue
        start = heading_match.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        block = text[start:end].strip()
        meta, text_lines = _parse_metadata_block(block)
        quoted = [ln.lstrip(">").strip() for ln in text_lines if ln.startswith(">")]
        prose = "\n".join(quoted) if quoted else " ".join(text_lines)
        needs_human_edit = "synthesized" in block.lower() and "not from a real letter" in block.lower()
        # e.g. stakeholder-collaboration-fragment in soft-skills-and-work-ethic.md:
        # a real block_id naming convention already distinguishes "-prose" blocks
        # (complete, safe to render standalone) from "-fragment" blocks (a phrase
        # meant to be worked into other prose, per that file's own usage guidance).
        is_fragment = block_id.endswith("-fragment")

        # Some files (greetings.md, closings.md) name each block after its
        # register directly instead of an explicit "**Register:**" line —
        # fall back to the heading itself when it's a known register slug.
        letter_register = meta.get("register") or (
            block_id if block_id in KNOWN_REGISTERS else None
        )
        extra_fields = {k: v for k, v in meta.items() if k != "register"}

        blocks.append(
            BuildingBlock(
                block_id=block_id,
                letter_register=letter_register,
                text=prose,
                needs_human_edit=needs_human_edit,
                is_fragment=is_fragment,
                extra_fields=extra_fields,
            )
        )

    return blocks


# ---------------------------------------------------------------------------
# Exclusion rules — Notes/skills-vault-status.md
# ---------------------------------------------------------------------------


def _load_exclusion_rules(
    vault_path: Path, warnings: list[str]
) -> tuple[list[str], list[str]]:
    excluded = list(DEFAULT_EXCLUDED_ASPIRATIONAL_SKILLS)
    forbidden = list(DEFAULT_FORBIDDEN_TERMS)

    status_path = vault_path / "Notes" / "skills-vault-status.md"
    if not status_path.is_file():
        warnings.append(
            f"{status_path} not found — falling back to hardcoded exclusion list {excluded}"
        )
        return excluded, forbidden

    text = status_path.read_text(encoding="utf-8")
    found = [name for name in DEFAULT_EXCLUDED_ASPIRATIONAL_SKILLS if name in text]
    if found:
        excluded = found
    else:
        warnings.append(
            f"could not confirm aspirational-skill exclusion list in {status_path} — "
            f"falling back to hardcoded default {excluded}"
        )

    return excluded, forbidden
