from pathlib import Path

from job_hunt_agent.core.vault_reader import load_vault


class TestLoadVault:
    def test_loads_three_variants(self, fixture_vault: Path):
        snap = load_vault(fixture_vault)
        assert set(snap.variants.keys()) == {"data-science", "bioinformatics", "ai-engineering"}

    def test_variant_frontmatter_fields(self, fixture_vault: Path):
        snap = load_vault(fixture_vault)
        ds = snap.variants["data-science"]
        assert ds.lane == "Data Science (generalist)"
        assert ds.subtitle == "Data Scientist | ML & Analytics"
        assert ds.reviewed is True
        assert ds.last_reviewed == "2026-01-01"
        assert ds.used_for_applications == []
        assert "test data scientist" in ds.summary_text.lower() or "data scientist" in ds.summary_text.lower()

    def test_ai_engineering_selected_projects(self, fixture_vault: Path):
        snap = load_vault(fixture_vault)
        ai = snap.variants["ai-engineering"]
        assert ai.has_selected_projects is True
        assert ai.selected_projects == ["test-project"]

    def test_variant_bullet_ids_used_parsed_from_comments(self, fixture_vault: Path):
        snap = load_vault(fixture_vault)
        ds = snap.variants["data-science"]
        assert "acme-ml-pipeline" in ds.bullet_ids_used
        assert "acme-dashboard" in ds.bullet_ids_used


class TestLoadExperience:
    def test_loads_employer_file(self, fixture_vault: Path):
        snap = load_vault(fixture_vault)
        assert "acme-corp" in snap.experience
        exp = snap.experience["acme-corp"]
        assert exp.employer == "Acme Corp"
        assert exp.dates == "2020 - 2024"

    def test_intro_notes_captured(self, fixture_vault: Path):
        snap = load_vault(fixture_vault)
        exp = snap.experience["acme-corp"]
        assert exp.intro_notes is not None
        assert "split across variants" in exp.intro_notes

    def test_bullet_used_in_parsed(self, fixture_vault: Path):
        snap = load_vault(fixture_vault)
        exp = snap.experience["acme-corp"]
        bullet = next(b for b in exp.bullets if b.bullet_id == "acme-ml-pipeline")
        assert bullet.used_in == ["data-science", "ai-engineering"]
        assert "machine learning pipelines" in bullet.text

    def test_bullet_relevance_note_captured(self, fixture_vault: Path):
        snap = load_vault(fixture_vault)
        exp = snap.experience["acme-corp"]
        bullet = next(b for b in exp.bullets if b.bullet_id == "acme-dashboard")
        assert bullet.relevance_note is not None
        assert "general-audience" in bullet.relevance_note

    def test_bullet_other_notes_captured_verbatim(self, fixture_vault: Path):
        snap = load_vault(fixture_vault)
        exp = snap.experience["acme-corp"]
        bullet = next(b for b in exp.bullets if b.bullet_id == "acme-genomics")
        assert any("kept only for the bioinformatics audience" in n for n in bullet.other_notes)

    def test_missing_used_in_produces_warning_not_crash(self, fixture_vault: Path):
        snap = load_vault(fixture_vault)
        exp = snap.experience["acme-corp"]
        bullet = next(b for b in exp.bullets if b.bullet_id == "acme-missing-used-in")
        assert bullet.used_in == []
        assert any("acme-missing-used-in" in w for w in snap.warnings)
        # the malformed block must not prevent the other bullets from loading
        assert len(exp.bullets) == 5

    def test_all_experience_bullets_helper(self, fixture_vault: Path):
        snap = load_vault(fixture_vault)
        assert len(snap.all_experience_bullets()) == 6

    def test_superseded_by_parsed_from_used_in_line(self, fixture_vault: Path):
        snap = load_vault(fixture_vault)
        exp = snap.experience["acme-corp"]
        original = next(b for b in exp.bullets if b.bullet_id == "acme-ml-pipeline-original")
        assert original.superseded_by == "acme-ml-pipeline"
        # unaffected bullets stay None, not accidentally matched
        modern = next(b for b in exp.bullets if b.bullet_id == "acme-ml-pipeline")
        assert modern.superseded_by is None


class TestLoadSkills:
    def test_loads_all_skill_files(self, fixture_vault: Path):
        snap = load_vault(fixture_vault)
        titles = {s.title for s in snap.skills}
        assert titles == {"ML Skills", "Bio Skills"}

    def test_used_entries_parsed_with_multiple_variants(self, fixture_vault: Path):
        snap = load_vault(fixture_vault)
        ml = next(s for s in snap.skills if s.title == "ML Skills")
        assert len(ml.used_entries) == 1
        entry = ml.used_entries[0]
        assert entry.keywords == ["Python", "scikit-learn"]
        assert set(entry.variants) == {"data-science", "ai-engineering"}

    def test_available_not_yet_used_detected(self, fixture_vault: Path):
        snap = load_vault(fixture_vault)
        ml = next(s for s in snap.skills if s.title == "ML Skills")
        assert "TensorFlow" in ml.available_not_yet_used
        assert "Deep learning" in ml.available_not_yet_used
        assert "Python" not in ml.available_not_yet_used

    def test_available_note_raw_captured_even_with_varied_wording(self, fixture_vault: Path):
        snap = load_vault(fixture_vault)
        bio = next(s for s in snap.skills if s.title == "Bio Skills")
        assert bio.available_note_raw is not None
        assert "nothing else" in bio.available_note_raw.lower()

    def test_category_path_reflects_subfolder(self, fixture_vault: Path):
        snap = load_vault(fixture_vault)
        ml = next(s for s in snap.skills if s.title == "ML Skills")
        assert "Category A" in ml.category_path


class TestLoadSharedComponents:
    def test_projects_parsed(self, fixture_vault: Path):
        snap = load_vault(fixture_vault)
        assert len(snap.projects) == 1
        assert snap.projects[0].entry_id == "test-project"
        assert snap.projects[0].used_in == ["ai-engineering"]

    def test_project_title_and_dates_captured(self, fixture_vault: Path):
        snap = load_vault(fixture_vault)
        project = snap.projects[0]
        assert project.title == "Test Agentic Pipeline"
        assert project.dates == "Personal engineering project, 2026"

    def test_project_text_is_only_bulleted_lines(self, fixture_vault: Path):
        snap = load_vault(fixture_vault)
        project = snap.projects[0]
        assert "Built a test agentic pipeline" in project.text
        assert "Delivered with a small mocked test suite" in project.text
        # internal provenance/reasoning must never leak into resume-facing text
        assert "internal test provenance note" not in project.text
        assert "Source" not in project.text

    def test_patents_parsed_and_status_heading_skipped(self, fixture_vault: Path):
        snap = load_vault(fixture_vault)
        ids = [p.entry_id for p in snap.patents]
        assert ids == ["test-patent"]
        assert snap.patents[0].used_in == ["data-science", "bioinformatics", "ai-engineering"]

    def test_patent_text_excludes_metadata_and_trailing_commentary(self, fixture_vault: Path):
        snap = load_vault(fixture_vault)
        patent = snap.patents[0]
        assert patent.text.strip() == "Co-inventor on a test patent for demonstration purposes"
        assert "internal test provenance note" not in patent.text
        assert "trailing commentary" not in patent.text

    def test_education_parsed(self, fixture_vault: Path):
        snap = load_vault(fixture_vault)
        assert len(snap.education) == 1
        assert snap.education[0].entry_id == "bs-example"


class TestLoadCoverLetters:
    def test_voice_example_parsed(self, fixture_vault: Path):
        snap = load_vault(fixture_vault)
        assert len(snap.voice_examples) == 1
        ex = snap.voice_examples[0]
        assert ex.company == "Acme Corp"
        assert "excited to apply" in ex.full_text

    def test_achievement_paragraph_parsed_with_quoted_prose(self, fixture_vault: Path):
        snap = load_vault(fixture_vault)
        assert len(snap.achievement_paragraphs) == 1
        para = snap.achievement_paragraphs[0]
        assert para.block_id == "acme-ml-prose"
        assert "built and deployed machine learning pipelines" in para.text.lower()

    def test_greetings_parsed(self, fixture_vault: Path):
        snap = load_vault(fixture_vault)
        assert len(snap.greetings) == 2

    def test_greeting_register_falls_back_to_heading(self, fixture_vault: Path):
        # greetings.md has no explicit "**Register:**" line — the heading
        # itself ("formal-professional") is the register.
        snap = load_vault(fixture_vault)
        greeting = next(g for g in snap.greetings if g.block_id == "formal-professional")
        assert greeting.letter_register == "formal-professional"

    def test_greeting_extra_fields_capture_salutation_and_opening_line(self, fixture_vault: Path):
        snap = load_vault(fixture_vault)
        greeting = next(g for g in snap.greetings if g.block_id == "formal-professional")
        assert "salutation" in greeting.extra_fields
        assert "opening line" in greeting.extra_fields
        assert "Dear Hiring Manager" in greeting.extra_fields["salutation"]

    def test_closings_parsed(self, fixture_vault: Path):
        snap = load_vault(fixture_vault)
        assert len(snap.closings) == 1

    def test_closing_register_from_explicit_meta_line(self, fixture_vault: Path):
        snap = load_vault(fixture_vault)
        assert snap.closings[0].letter_register == "formal-professional"

    def test_soft_skills_needs_human_edit_flagged(self, fixture_vault: Path):
        snap = load_vault(fixture_vault)
        synthesized = next(
            b for b in snap.soft_skills if b.block_id == "synthesized-leadership-draft"
        )
        assert synthesized.needs_human_edit is True
        verbatim = next(b for b in snap.soft_skills if b.block_id == "easy-to-work-with-prose")
        assert verbatim.needs_human_edit is False

    def test_fragment_suffixed_block_id_flagged_as_fragment(self, fixture_vault: Path):
        snap = load_vault(fixture_vault)
        fragment = next(
            b for b in snap.soft_skills if b.block_id == "stakeholder-collaboration-fragment"
        )
        assert fragment.is_fragment is True
        prose = next(b for b in snap.soft_skills if b.block_id == "easy-to-work-with-prose")
        assert prose.is_fragment is False


class TestExclusionRules:
    def test_excluded_aspirational_skills_found_in_notes(self, fixture_vault: Path):
        snap = load_vault(fixture_vault)
        assert set(snap.excluded_aspirational_skills) == {
            "CrewAI",
            "RAG",
            "Finetuning",
            "Hugging Face",
        }

    def test_forbidden_terms_always_includes_fda(self, fixture_vault: Path):
        snap = load_vault(fixture_vault)
        assert "FDA" in snap.forbidden_terms

    def test_falls_back_when_notes_file_missing(self, tmp_path: Path, fixture_vault: Path):
        # copy the fixture vault but delete Notes/skills-vault-status.md
        (fixture_vault / "Notes" / "skills-vault-status.md").unlink()
        snap = load_vault(fixture_vault)
        assert snap.excluded_aspirational_skills == ["CrewAI", "RAG", "Finetuning", "Hugging Face"]
        assert any("skills-vault-status.md" in w for w in snap.warnings)


class TestRobustness:
    def test_missing_directories_produce_warnings_not_crash(self, tmp_path: Path):
        empty_vault = tmp_path / "empty-vault"
        empty_vault.mkdir()
        snap = load_vault(empty_vault)
        assert snap.variants == {}
        assert snap.experience == {}
        assert snap.skills == []
        assert len(snap.warnings) > 0
