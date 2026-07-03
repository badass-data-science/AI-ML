from pathlib import Path

import pytest

from job_hunt_agent.core.assembler import (
    assemble_draft_cover_letter,
    assemble_draft_resume,
    slugify,
)
from job_hunt_agent.core.models import JobMatchResult
from job_hunt_agent.core.vault_reader import load_vault


@pytest.fixture
def loaded_vault(fixture_vault: Path):
    return load_vault(fixture_vault)


@pytest.fixture
def match_result(sample_job_posting, sample_llm_output):
    return JobMatchResult(job=sample_job_posting, llm_output=sample_llm_output)


class TestSlugify:
    def test_basic_slug(self):
        assert slugify("Acme Corp", "Data Scientist", "2026-01-01") == "acme-corp-data-scientist-2026-01-01"

    def test_empty_parts_dropped(self):
        assert slugify("Acme", "", "role") == "acme-role"

    def test_falls_back_when_empty(self):
        assert slugify("", "") == "untitled"


class TestAssembleDraftResume:
    def test_raises_without_llm_output(self, loaded_vault, sample_job_posting, tmp_path):
        empty_match = JobMatchResult(job=sample_job_posting, llm_output=None)
        with pytest.raises(ValueError):
            assemble_draft_resume(empty_match, loaded_vault, tmp_path)

    def test_writes_resume_md(self, loaded_vault, match_result, tmp_path):
        draft = assemble_draft_resume(match_result, loaded_vault, tmp_path)
        assert draft.output_path.exists()
        assert draft.output_path.name == "resume.md"
        assert draft.variant == "data-science"

    def test_resume_contains_variant_content(self, loaded_vault, match_result, tmp_path):
        draft = assemble_draft_resume(match_result, loaded_vault, tmp_path)
        text = draft.output_path.read_text()
        assert "Data Scientist | ML & Analytics" in text
        assert "machine learning" in text.lower()

    def test_variant_override_used(self, loaded_vault, match_result, tmp_path):
        draft = assemble_draft_resume(match_result, loaded_vault, tmp_path, variant_override="ai-engineering")
        assert draft.variant == "ai-engineering"
        text = draft.output_path.read_text()
        assert "AI Engineer" in text

    def test_unknown_variant_override_raises(self, loaded_vault, match_result, tmp_path):
        with pytest.raises(ValueError):
            assemble_draft_resume(match_result, loaded_vault, tmp_path, variant_override="not-real")

    def test_surfaced_content_marked_as_new(self, loaded_vault, match_result, tmp_path):
        draft = assemble_draft_resume(match_result, loaded_vault, tmp_path, include_surfaced=True)
        text = draft.output_path.read_text()
        assert "NEW: surfaced by matcher" in text
        assert "why relevant" in text.lower()

    def test_surfaced_content_excluded_when_disabled(self, loaded_vault, match_result, tmp_path):
        draft = assemble_draft_resume(match_result, loaded_vault, tmp_path, include_surfaced=False)
        text = draft.output_path.read_text()
        assert "NEW: surfaced by matcher" not in text

    def test_surfaced_bullet_already_in_variant_is_dropped(
        self, loaded_vault, sample_job_posting, sample_llm_output, tmp_path
    ):
        # Regression test for a real bug found during manual verification: the
        # LLM sometimes claims a bullet is "surfaced" (implying unused) when
        # the vault's own `used_in` tag says it's already in the recommended
        # variant's Experience section. acme-ml-pipeline is used_in
        # data-science (the recommended variant here) — it must never be
        # rendered a second time under "Surfaced bullets".
        from job_hunt_agent.core.models import SurfacedBullet

        sample_llm_output.surfaced_bullets = [
            SurfacedBullet(
                bullet_id="acme-ml-pipeline", employer="Acme Corp", why_relevant="duplicate claim"
            )
        ]
        match = JobMatchResult(job=sample_job_posting, llm_output=sample_llm_output)
        draft = assemble_draft_resume(match, loaded_vault, tmp_path)
        text = draft.output_path.read_text()
        assert "NEW: surfaced by matcher" not in text
        # the bullet's real text still appears exactly once, in Experience
        assert text.count("Built and deployed machine learning pipelines") == 1

    def test_surfaced_skill_already_in_variant_is_dropped(
        self, loaded_vault, sample_job_posting, sample_llm_output, tmp_path
    ):
        # Same regression as above, for skills: "Python" is already used_in
        # data-science per the fixture vault's ML Skills.md.
        from job_hunt_agent.core.models import SurfacedSkill

        sample_llm_output.surfaced_skills = [
            SurfacedSkill(file_title="ML Skills", keyword="Python", why_relevant="duplicate claim"),
            SurfacedSkill(
                file_title="ML Skills", keyword="TensorFlow", why_relevant="genuinely unused"
            ),
        ]
        match = JobMatchResult(job=sample_job_posting, llm_output=sample_llm_output)
        draft = assemble_draft_resume(match, loaded_vault, tmp_path)
        text = draft.output_path.read_text()
        assert "Surfaced skills not yet in this resume" in text
        assert "TensorFlow" in text
        # Python must not appear in the surfaced-skills section specifically
        surfaced_section = text.split("Surfaced skills not yet in this resume")[1].split("##")[0]
        assert "Python" not in surfaced_section

    def test_surfaced_skill_rendered_via_skills_section_raw_but_untracked_is_dropped(
        self, loaded_vault, sample_job_posting, sample_llm_output, tmp_path
    ):
        # Regression test for a real bug: the dedup check above only looks at
        # each Skills file's structured `used_entries` tags, not the resume's
        # actual rendered Skills text. A vault can tag an abbreviation as used
        # ("MCP") while the resume's skills_section_raw renders a synonym
        # alongside it ("Model Context Protocol (MCP)") that no used_entries
        # tag ever names — the old check missed that gap entirely. The fixture
        # vault has the same shape: data-science's skills_section_raw is
        # "Python | R | SQL", but neither "R" nor "SQL" is tracked in any
        # Skills file's used_entries — so the old code would have surfaced
        # "SQL" as unused despite it being right there in the resume already.
        from job_hunt_agent.core.models import SurfacedSkill

        sample_llm_output.surfaced_skills = [
            SurfacedSkill(file_title="ML Skills", keyword="SQL", why_relevant="duplicate claim"),
            SurfacedSkill(
                file_title="ML Skills", keyword="TensorFlow", why_relevant="genuinely unused"
            ),
        ]
        match = JobMatchResult(job=sample_job_posting, llm_output=sample_llm_output)
        draft = assemble_draft_resume(match, loaded_vault, tmp_path)
        text = draft.output_path.read_text()
        assert "TensorFlow" in text
        surfaced_section = text.split("Surfaced skills not yet in this resume")[1].split("##")[0]
        assert "SQL" not in surfaced_section

    def test_projects_only_included_for_variant_with_selected_projects(
        self, loaded_vault, match_result, tmp_path
    ):
        # data-science variant has has_selected_projects: false
        draft = assemble_draft_resume(match_result, loaded_vault, tmp_path)
        text = draft.output_path.read_text()
        assert "Selected Projects" not in text

    def test_guardrail_violation_still_writes_but_flags_warning(
        self, loaded_vault, sample_job_posting, sample_llm_output, tmp_path
    ):
        # tamper the summary so the assembled draft contains a forbidden term
        loaded_vault.variants["data-science"].summary_text += " Supported FDA approval."
        match = JobMatchResult(job=sample_job_posting, llm_output=sample_llm_output)
        draft = assemble_draft_resume(match, loaded_vault, tmp_path)
        assert draft.output_path.exists()
        assert any("FDA" in w for w in draft.warnings)
        # text is NOT silently stripped
        assert "FDA" in draft.output_path.read_text()

    def test_source_url_included_in_metadata_comment_when_present(
        self, loaded_vault, sample_job_posting, sample_llm_output, tmp_path
    ):
        job = sample_job_posting.model_copy(update={"url": "https://acme.com/careers/123"})
        match = JobMatchResult(job=job, llm_output=sample_llm_output)
        draft = assemble_draft_resume(match, loaded_vault, tmp_path)
        assert "Source posting: https://acme.com/careers/123" in draft.output_path.read_text()

    def test_no_source_url_line_when_absent(self, loaded_vault, match_result, tmp_path):
        draft = assemble_draft_resume(match_result, loaded_vault, tmp_path)
        assert "Source posting:" not in draft.output_path.read_text()


class TestAssembleDraftCoverLetter:
    def test_raises_without_llm_output(self, loaded_vault, sample_job_posting, tmp_path):
        empty_match = JobMatchResult(job=sample_job_posting, llm_output=None)
        with pytest.raises(ValueError):
            assemble_draft_cover_letter(empty_match, loaded_vault, tmp_path)

    def test_writes_cover_letter_md(self, loaded_vault, match_result, tmp_path):
        draft = assemble_draft_cover_letter(match_result, loaded_vault, tmp_path)
        assert draft.output_path.exists()
        assert draft.output_path.name == "cover_letter.md"
        assert draft.letter_register == "formal-professional"

    def test_company_specific_placeholder_always_present(self, loaded_vault, match_result, tmp_path):
        draft = assemble_draft_cover_letter(match_result, loaded_vault, tmp_path)
        text = draft.output_path.read_text()
        assert "WRITE COMPANY-SPECIFIC PARAGRAPH HERE" in text

    def test_achievement_paragraph_included(self, loaded_vault, match_result, tmp_path):
        draft = assemble_draft_cover_letter(match_result, loaded_vault, tmp_path)
        text = draft.output_path.read_text()
        assert "machine learning pipelines end to end" in text.lower()

    def test_synthesized_soft_skill_gets_warning_comment(
        self, loaded_vault, sample_job_posting, sample_llm_output, tmp_path
    ):
        sample_llm_output.recommended_soft_skill_id = "synthesized-leadership-draft"
        match = JobMatchResult(job=sample_job_posting, llm_output=sample_llm_output)
        draft = assemble_draft_cover_letter(match, loaded_vault, tmp_path)
        text = draft.output_path.read_text()
        assert "flagged in the vault as synthesized" in text

    def test_register_override_used(self, loaded_vault, match_result, tmp_path):
        draft = assemble_draft_cover_letter(match_result, loaded_vault, tmp_path, register_override="casual-direct")
        assert draft.letter_register == "casual-direct"

    def test_salutation_has_no_raw_template_markup(self, loaded_vault, match_result, tmp_path):
        # greetings.md's real content wraps salutation/opening-line values in
        # backticks (and sometimes lists " / "-separated alternates) — the
        # assembled draft should read as plain text, not leak that
        # markdown-template formatting verbatim.
        draft = assemble_draft_cover_letter(match_result, loaded_vault, tmp_path)
        text = draft.output_path.read_text()
        assert "`" not in text

    def test_source_url_included_in_metadata_comment_when_present(
        self, loaded_vault, sample_job_posting, sample_llm_output, tmp_path
    ):
        job = sample_job_posting.model_copy(update={"url": "https://acme.com/careers/123"})
        match = JobMatchResult(job=job, llm_output=sample_llm_output)
        draft = assemble_draft_cover_letter(match, loaded_vault, tmp_path)
        assert "Source posting: https://acme.com/careers/123" in draft.output_path.read_text()

    def test_no_source_url_line_when_absent(self, loaded_vault, match_result, tmp_path):
        draft = assemble_draft_cover_letter(match_result, loaded_vault, tmp_path)
        assert "Source posting:" not in draft.output_path.read_text()


class TestNoInternalReasoningLeaksIntoDrafts:
    """Regression coverage for a real bug caught during manual end-to-end
    verification: internal vault provenance/reasoning text (patents.md's
    **Source:** lines, trailing commentary paragraphs) was leaking verbatim
    into resume-facing drafts."""

    def test_projects_and_patents_sections_exclude_internal_provenance(
        self, loaded_vault, sample_job_posting, sample_llm_output, tmp_path
    ):
        sample_llm_output.recommended_variant = "ai-engineering"
        match = JobMatchResult(job=sample_job_posting, llm_output=sample_llm_output)
        draft = assemble_draft_resume(match, loaded_vault, tmp_path)
        text = draft.output_path.read_text()
        assert "internal test provenance note" not in text
        assert "trailing commentary" not in text
        # the actual resume-facing bullet content must still be present
        assert "Built a test agentic pipeline" in text
        assert "Co-inventor on a test patent" in text
