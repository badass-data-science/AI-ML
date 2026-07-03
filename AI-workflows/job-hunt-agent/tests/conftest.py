"""Shared fixtures. All vault fixtures build a tiny synthetic vault under
tmp_path, mirroring vault-Resume's real conventions — tests never touch the
real vault-Resume directory.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from job_hunt_agent.core.models import (
    JobMatchLLMOutput,
    JobPosting,
    SurfacedBullet,
    SurfacedSkill,
    TokenUsage,
    VariantScore,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture
def fixture_vault(tmp_path: Path) -> Path:
    """A minimal synthetic vault-Resume, mirroring the real conventions."""
    vault = tmp_path / "vault-Resume"

    _write(
        vault / "Resumes" / "variants" / "data-science.md",
        """---
variant: data-science
lane: Data Science (generalist)
subtitle: "Data Scientist | ML & Analytics"
audience: General data science roles
has_selected_projects: false
education_order: [bs-example]
provenance: "test fixture"
reviewed: true
last_reviewed: "2026-01-01"
converted_to_docx_or_pdf: false
used_for_applications: []
---

# Test Person

**Data Scientist | ML & Analytics**
Somewhere, USA | test@example.com

## Professional Summary

A data scientist with test experience in machine learning and statistics.

## Skills

**Programming:** Python | R | SQL

## Experience

**Data Scientist | Acme Corp**
*2020 - 2024*
<!-- bullets: acme-ml-pipeline, acme-dashboard -->
- Built ML pipelines
- Built dashboards

## Education

**B.S. Example** | Test University
""",
    )

    _write(
        vault / "Resumes" / "variants" / "ai-engineering.md",
        """---
variant: ai-engineering
lane: AI Engineering
subtitle: "AI Engineer | Agentic Systems"
audience: AI/LLM engineering roles
has_selected_projects: true
selected_projects: [test-project]
education_order: [bs-example]
provenance: "test fixture"
reviewed: true
last_reviewed: "2026-01-01"
converted_to_docx_or_pdf: false
used_for_applications: []
---

# Test Person

**AI Engineer | Agentic Systems**

## Professional Summary

An AI engineer with test agentic pipeline experience.

## Skills

**AI Engineering:** litellm | instructor | Pydantic

## Experience

**Data Scientist | Acme Corp**
*2020 - 2024*
<!-- bullets: acme-ml-pipeline -->
- Built ML pipelines
""",
    )

    _write(
        vault / "Resumes" / "variants" / "bioinformatics.md",
        """---
variant: bioinformatics
lane: Bioinformatics
subtitle: "Bioinformatics Data Scientist"
audience: Biotech roles
has_selected_projects: false
education_order: [bs-example]
provenance: "test fixture"
reviewed: true
last_reviewed: "2026-01-01"
converted_to_docx_or_pdf: false
used_for_applications: []
---

# Test Person

**Bioinformatics Data Scientist**

## Professional Summary

A bioinformatics data scientist with test genomics experience.

## Skills

**Bioinformatics:** NGS | CRISPR

## Experience

**Data Scientist | Acme Corp**
*2020 - 2024*
<!-- bullets: acme-genomics -->
- Built genomics pipelines
""",
    )

    _write(
        vault / "Resumes" / "experience" / "acme-corp.md",
        """---
employer: Acme Corp
title: Data Scientist
dates: 2020 - 2024
---

# Acme Corp — Data Scientist
*2020 - 2024*

This employer has bullets split across variants for testing purposes.

## Bullets

### acme-ml-pipeline
**Used in:** data-science, ai-engineering
Built and deployed machine learning pipelines using Python and scikit-learn

### acme-dashboard
**Used in:** data-science
**Relevance:** general-audience dashboard work, not bio-specific.
Built an interactive dashboard for stakeholders using Plotly and Flask

### acme-genomics
**Used in:** bioinformatics
**Note:** kept only for the bioinformatics audience.
Built genomics data pipelines for variant calling using Python and R

### acme-missing-used-in
Bullet intentionally missing a Used in line, to test the warning path

### acme-ml-pipeline-original
**Used in:** none (retired — superseded by `acme-ml-pipeline`)
Built machine learning pipelines using Python and scikit-learn for internal reporting
""",
    )

    _write(
        vault / "Resumes" / "experience" / "zenith-corp.md",
        """---
employer: Zenith Corp
title: Data Scientist
dates: 2024 - 2026
---

# Zenith Corp — Data Scientist
*2024 - 2026*

Deliberately named/filed to sort alphabetically *after* acme-corp.md
(z > a) despite being dated *more recently*, to test that Experience
ordering is chronological and not just alphabetical file-loading order.

## Bullets

### zenith-recent-work
**Used in:** data-science
Did recent data science work at Zenith
""",
    )

    _write(
        vault / "Resumes" / "projects.md",
        """# Selected Projects Library

## test-project

**Used in:** ai-engineering
**Title:** Test Agentic Pipeline
**Dates:** Personal engineering project, 2026
**Source:** internal test provenance note — must never leak into a rendered draft

- Built a test agentic pipeline with structured outputs and async orchestration
- Delivered with a small mocked test suite
""",
    )

    _write(
        vault / "Resumes" / "patents-and-publications.md",
        """# Patents & Publications

## test-patent

**Used in:** data-science, bioinformatics, ai-engineering
**Title:** Test Patent Title
**Source:** internal test provenance note — must never leak into a rendered draft

- Co-inventor on a test patent for demonstration purposes

**This is trailing commentary that must also never leak into a rendered draft.**

## Status

Both entries kept identical across variants.
""",
    )

    _write(
        vault / "Resumes" / "education.md",
        """# Education

Same entries in every variant.

## Entries

### bs-example
**B.S. Example** | Test University
Completed coursework in testing.

## Ordering per variant

All variants use the same order.
""",
    )

    _write(
        vault / "Skills" / "Category A" / "ML Skills.md",
        """# ML Skills

## Used in current resumes

* Python, scikit-learn → **data-science** and **ai-engineering variants**
* Everything else in this file: available, not yet used in any resume

## Core

* Python
* scikit-learn
* TensorFlow
* Deep learning
""",
    )

    _write(
        vault / "Skills" / "Category A" / "Bio Skills.md",
        """# Bio Skills

## Used in current resumes

* NGS, CRISPR → **bioinformatics variant**
* Nothing else in this file is used in any other variant currently.

## Genomics

* NGS
* CRISPR
* Variant calling
* Single-cell genomics
""",
    )

    _write(
        vault / "CoverLetters" / "voice-examples" / "acme-letter.md",
        """---
company: Acme Corp
role: Data Scientist
register: formal-professional
---

# Acme Corp — Data Scientist

## Full text

Dear Hiring Manager,

I am excited to apply.

Sincerely,
Test Person

## Notes

A test voice example.
""",
    )

    _write(
        vault / "CoverLetters" / "building-blocks" / "achievement-paragraphs.md",
        """# Achievement Paragraphs

## acme-ml-prose

**Source:** Acme letter
**Maps to:** [[acme-corp#acme-ml-pipeline|acme-ml-pipeline]]
**Register:** formal-professional

> At Acme Corp, I built and deployed machine learning pipelines end to end.
""",
    )

    _write(
        vault / "CoverLetters" / "building-blocks" / "greetings.md",
        """# Greeting & Opening Patterns

## formal-professional

**Used in:** Acme Corp (Data Scientist)
**Salutation:** `Dear Hiring Manager,`
**Opening line:** `I am excited to apply for the [ROLE] position at [COMPANY].`

## casual-direct

**Used in:** nothing in this fixture
**Salutation:** `Hi,`
**Opening line:** `Hi, my name is Test Person.`
""",
    )

    _write(
        vault / "CoverLetters" / "building-blocks" / "closings.md",
        """# Closings

## standard-closing

**Register:** formal-professional

> Sincerely, Test Person
""",
    )

    _write(
        vault / "CoverLetters" / "building-blocks" / "soft-skills-and-work-ethic.md",
        """# Soft Skills and Work Ethic

## easy-to-work-with-prose

**Source:** Acme letter
**Register:** formal-professional

> I am easy to work with and a clear communicator.

## synthesized-leadership-draft

**Register:** formal-professional

> ⚠️ synthesized, not from a real letter — review before using. Draft leadership paragraph.

## stakeholder-collaboration-fragment

**Register:** formal-professional

> ...strong experience in error analysis, data quality, and stakeholder collaboration.
""",
    )

    _write(
        vault / "Notes" / "skills-vault-status.md",
        """# Skills Vault Status (test fixture)

Explicitly excluded, not an oversight: CrewAI, RAG, Finetuning, Hugging Face —
aspirational, not yet real skills.
""",
    )

    return vault


@pytest.fixture
def sample_job_posting() -> JobPosting:
    return JobPosting(
        raw_text="We are looking for a Data Scientist skilled in Python and machine learning.",
        source_path=None,
        company="Acme Corp",
        role_title="Data Scientist",
        fetched_at=datetime(2026, 1, 1),
    )


@pytest.fixture
def sample_llm_output() -> JobMatchLLMOutput:
    return JobMatchLLMOutput(
        variant_scores=[
            VariantScore(
                variant="data-science",
                fit_score=0.9,
                reasoning=["Strong keyword overlap", "Matches core DS skillset"],
                strengths=["Python", "ML pipelines"],
                gaps=[],
            ),
            VariantScore(
                variant="bioinformatics",
                fit_score=0.2,
                reasoning=["Posting has no bio/genomics content", "Low relevance"],
                strengths=["General statistical background still applies"],
                gaps=["No bio-specific evidence needed"],
            ),
            VariantScore(
                variant="ai-engineering",
                fit_score=0.5,
                reasoning=["Some ML overlap", "Not agentic-specific"],
                strengths=["ML pipelines"],
                gaps=["No agentic/LLM requirement in posting"],
            ),
        ],
        recommended_variant="data-science",
        cover_letter_register="formal-professional",
        register_reasoning="Posting reads as a standard corporate role.",
        surfaced_skills=[
            SurfacedSkill(
                file_title="ML Skills", keyword="TensorFlow", why_relevant="ML-adjacent"
            )
        ],
        surfaced_bullets=[
            SurfacedBullet(
                # acme-genomics is used_in: bioinformatics only — genuinely
                # absent from the data-science variant's Experience section,
                # unlike acme-dashboard (used_in: data-science) which would
                # have been a duplicate, not a real "surfaced" example.
                bullet_id="acme-genomics",
                employer="Acme Corp",
                why_relevant="Shows structured data pipeline experience",
            )
        ],
        recommended_achievement_paragraph_ids=["acme-ml-prose"],
        recommended_soft_skill_id="easy-to-work-with-prose",
    )


@pytest.fixture
def sample_token_usage() -> TokenUsage:
    return TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
