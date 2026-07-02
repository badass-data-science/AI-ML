from pathlib import Path

from job_hunt_agent.core.guardrails import scan_for_violations
from job_hunt_agent.core.vault_reader import load_vault


class TestScanForViolations:
    def test_clean_text_has_no_violations(self, fixture_vault: Path):
        vault = load_vault(fixture_vault)
        violations = scan_for_violations("A clean resume with Python and SQL.", vault)
        assert violations == []

    def test_fda_is_always_caught(self, fixture_vault: Path):
        vault = load_vault(fixture_vault)
        violations = scan_for_violations("Supported FDA approval of a diagnostic assay.", vault)
        assert any("FDA" in v for v in violations)

    def test_fda_caught_case_insensitively(self, fixture_vault: Path):
        vault = load_vault(fixture_vault)
        violations = scan_for_violations("supported fda approval", vault)
        assert any("FDA" in v for v in violations)

    def test_each_excluded_skill_is_caught(self, fixture_vault: Path):
        vault = load_vault(fixture_vault)
        for skill in ["CrewAI", "RAG", "Finetuning", "Hugging Face"]:
            violations = scan_for_violations(f"Experience with {skill} pipelines.", vault)
            assert any(skill in v for v in violations), f"{skill} was not caught"

    def test_excluded_skill_caught_case_insensitively(self, fixture_vault: Path):
        vault = load_vault(fixture_vault)
        violations = scan_for_violations("built agents with crewai", vault)
        assert any("CrewAI" in v for v in violations)

    def test_multiple_violations_all_reported(self, fixture_vault: Path):
        vault = load_vault(fixture_vault)
        violations = scan_for_violations("FDA approval using RAG and CrewAI.", vault)
        assert len(violations) == 3

    def test_short_acronym_does_not_false_positive_inside_ordinary_words(self, fixture_vault: Path):
        # "RAG" is a substring of "paragraph" and "leveraging" — a naive
        # substring check would wrongly flag these. Regression test for a
        # real false positive hit during manual end-to-end verification.
        vault = load_vault(fixture_vault)
        violations = scan_for_violations(
            "Write a company-specific paragraph leveraging your experience.", vault
        )
        assert violations == []

    def test_rag_still_caught_as_its_own_word(self, fixture_vault: Path):
        vault = load_vault(fixture_vault)
        violations = scan_for_violations("Built a RAG pipeline.", vault)
        assert any("RAG" in v for v in violations)
