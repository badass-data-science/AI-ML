from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from job_hunt_agent.cli import app
from job_hunt_agent.core.models import JobMatchResult, TokenUsage

runner = CliRunner()


def _mock_llm_client_class(llm_output):
    """Patch target: job_hunt_agent.cli.LLMClient — the name as imported into
    the CLI module's namespace, since that's what match_cmd actually calls
    when constructing a client for a real run.
    """
    mock_instance = MagicMock()
    mock_instance.complete_structured = AsyncMock(
        return_value=(llm_output, TokenUsage(total_tokens=10))
    )
    return MagicMock(return_value=mock_instance)


class TestLoadVaultCmd:
    def test_smoke_test_against_fixture_vault(self, fixture_vault: Path):
        result = runner.invoke(app, ["load-vault", "--vault-path", str(fixture_vault)])
        assert result.exit_code == 0
        assert "ai-engineering" in result.stdout
        # the fixture vault deliberately has one malformed bullet (see conftest.py)
        # to exercise the warning path, so a warning is expected here, not absent.
        assert "Warnings (1):" in result.stdout
        assert "acme-missing-used-in" in result.stdout


class TestMatchCmd:
    def test_writes_match_json_and_prints_summary(
        self, fixture_vault: Path, sample_llm_output, tmp_path: Path
    ):
        posting_file = tmp_path / "posting.txt"
        posting_file.write_text("We need a Data Scientist skilled in Python and ML.")

        with patch("job_hunt_agent.cli.LLMClient", _mock_llm_client_class(sample_llm_output)):
            result = runner.invoke(
                app,
                [
                    "match",
                    "--posting", str(posting_file),
                    "--company", "Acme",
                    "--role", "Data Scientist",
                    "--vault-path", str(fixture_vault),
                    "--home", str(tmp_path),
                ],
            )

        assert result.exit_code == 0, result.stdout
        assert "Recommended variant: data-science" in result.stdout
        match_files = list((tmp_path / "output" / "matches").rglob("match.json"))
        assert len(match_files) == 1
        saved = JobMatchResult.model_validate_json(match_files[0].read_text())
        assert saved.llm_output.recommended_variant == "data-science"

    def test_url_flag_carried_into_saved_match_json(
        self, fixture_vault: Path, sample_llm_output, tmp_path: Path
    ):
        posting_file = tmp_path / "posting.txt"
        posting_file.write_text("We need a Data Scientist skilled in Python and ML.")

        with patch("job_hunt_agent.cli.LLMClient", _mock_llm_client_class(sample_llm_output)):
            result = runner.invoke(
                app,
                [
                    "match",
                    "--posting", str(posting_file),
                    "--company", "Acme",
                    "--role", "Data Scientist",
                    "--url", "https://acme.com/careers/123",
                    "--vault-path", str(fixture_vault),
                    "--home", str(tmp_path),
                ],
            )

        assert result.exit_code == 0, result.stdout
        match_files = list((tmp_path / "output" / "matches").rglob("match.json"))
        saved = JobMatchResult.model_validate_json(match_files[0].read_text())
        assert saved.job.url == "https://acme.com/careers/123"

    def test_reads_posting_from_stdin(self, fixture_vault: Path, sample_llm_output, tmp_path: Path):
        with patch("job_hunt_agent.cli.LLMClient", _mock_llm_client_class(sample_llm_output)):
            result = runner.invoke(
                app,
                [
                    "match",
                    "--posting", "-",
                    "--vault-path", str(fixture_vault),
                    "--home", str(tmp_path),
                ],
                input="Data Scientist role requiring Python.",
            )
        assert result.exit_code == 0, result.stdout

    def test_missing_posting_file_errors_cleanly(self, fixture_vault: Path, tmp_path: Path):
        result = runner.invoke(
            app,
            [
                "match",
                "--posting", str(tmp_path / "does-not-exist.txt"),
                "--vault-path", str(fixture_vault),
                "--home", str(tmp_path),
            ],
        )
        assert result.exit_code != 0


class TestDraftCmd:
    def _write_match_json(self, tmp_path: Path, sample_job_posting, sample_llm_output) -> Path:
        match = JobMatchResult(job=sample_job_posting, llm_output=sample_llm_output)
        match_path = tmp_path / "match.json"
        match_path.write_text(match.model_dump_json(indent=2), encoding="utf-8")
        return match_path

    def test_draft_from_saved_match(
        self, fixture_vault: Path, sample_job_posting, sample_llm_output, tmp_path: Path
    ):
        match_path = self._write_match_json(tmp_path, sample_job_posting, sample_llm_output)
        output_dir = tmp_path / "draft-out"

        result = runner.invoke(
            app,
            [
                "draft",
                "--match", str(match_path),
                "--vault-path", str(fixture_vault),
                "--output-dir", str(output_dir),
            ],
        )

        assert result.exit_code == 0, result.stdout
        assert (output_dir / "resume.md").exists()
        assert (output_dir / "cover_letter.md").exists()

    def test_draft_respects_variant_override(
        self, fixture_vault: Path, sample_job_posting, sample_llm_output, tmp_path: Path
    ):
        match_path = self._write_match_json(tmp_path, sample_job_posting, sample_llm_output)
        output_dir = tmp_path / "draft-out"

        result = runner.invoke(
            app,
            [
                "draft",
                "--match", str(match_path),
                "--variant", "ai-engineering",
                "--vault-path", str(fixture_vault),
                "--output-dir", str(output_dir),
            ],
        )

        assert result.exit_code == 0, result.stdout
        assert "ai-engineering" in result.stdout


class TestInitFilledCmd:
    def test_creates_filled_copies_of_both_drafts(self, tmp_path: Path):
        (tmp_path / "resume.md").write_text("# Resume draft", encoding="utf-8")
        (tmp_path / "cover_letter.md").write_text("Dear Hiring Manager,", encoding="utf-8")

        result = runner.invoke(app, ["init-filled", "--draft-dir", str(tmp_path)])

        assert result.exit_code == 0, result.stdout
        assert (tmp_path / "resume-filled.md").read_text() == "# Resume draft"
        assert (tmp_path / "cover_letter-filled.md").read_text() == "Dear Hiring Manager,"
        assert "Created" in result.stdout

    def test_only_creates_filled_copy_for_files_that_exist(self, tmp_path: Path):
        (tmp_path / "resume.md").write_text("# Resume draft", encoding="utf-8")

        result = runner.invoke(app, ["init-filled", "--draft-dir", str(tmp_path)])

        assert result.exit_code == 0, result.stdout
        assert (tmp_path / "resume-filled.md").exists()
        assert not (tmp_path / "cover_letter-filled.md").exists()

    def test_does_not_overwrite_existing_filled_file_without_force(self, tmp_path: Path):
        (tmp_path / "resume.md").write_text("# Regenerated draft", encoding="utf-8")
        (tmp_path / "resume-filled.md").write_text("# Hand-edited content, do not lose", encoding="utf-8")

        result = runner.invoke(app, ["init-filled", "--draft-dir", str(tmp_path)])

        assert result.exit_code == 0, result.stdout
        assert (tmp_path / "resume-filled.md").read_text() == "# Hand-edited content, do not lose"
        assert "left untouched" in result.stdout

    def test_force_overwrites_existing_filled_file(self, tmp_path: Path):
        (tmp_path / "resume.md").write_text("# Regenerated draft", encoding="utf-8")
        (tmp_path / "resume-filled.md").write_text("# Stale hand-edited content", encoding="utf-8")

        result = runner.invoke(app, ["init-filled", "--draft-dir", str(tmp_path), "--force"])

        assert result.exit_code == 0, result.stdout
        assert (tmp_path / "resume-filled.md").read_text() == "# Regenerated draft"

    def test_no_draft_files_present_errors(self, tmp_path: Path):
        result = runner.invoke(app, ["init-filled", "--draft-dir", str(tmp_path)])
        assert result.exit_code == 1


class TestMatchAndDraftCmd:
    def test_chains_match_and_draft(
        self, fixture_vault: Path, sample_llm_output, tmp_path: Path
    ):
        posting_file = tmp_path / "posting.txt"
        posting_file.write_text("Data Scientist role requiring Python and ML.")

        with patch("job_hunt_agent.cli.LLMClient", _mock_llm_client_class(sample_llm_output)):
            result = runner.invoke(
                app,
                [
                    "match-and-draft",
                    "--posting", str(posting_file),
                    "--company", "Acme",
                    "--role", "Data Scientist",
                    "--vault-path", str(fixture_vault),
                    "--home", str(tmp_path),
                ],
            )

        assert result.exit_code == 0, result.stdout
        assert list((tmp_path / "output" / "matches").rglob("match.json"))
        assert list((tmp_path / "output" / "drafts").rglob("resume.md"))
        assert list((tmp_path / "output" / "drafts").rglob("cover_letter.md"))

    def _invoke_match_and_draft(self, fixture_vault, sample_llm_output, tmp_path, extra_args=()):
        posting_file = tmp_path / "posting.txt"
        posting_file.write_text("Data Scientist role requiring Python and ML.")
        with patch("job_hunt_agent.cli.LLMClient", _mock_llm_client_class(sample_llm_output)):
            return runner.invoke(
                app,
                [
                    "match-and-draft",
                    "--posting", str(posting_file),
                    "--company", "Acme",
                    "--role", "Data Scientist",
                    "--vault-path", str(fixture_vault),
                    "--home", str(tmp_path),
                    *extra_args,
                ],
            )

    def test_creates_filled_copies_by_default(
        self, fixture_vault: Path, sample_llm_output, tmp_path: Path
    ):
        result = self._invoke_match_and_draft(fixture_vault, sample_llm_output, tmp_path)
        assert result.exit_code == 0, result.stdout
        assert list((tmp_path / "output" / "drafts").rglob("resume-filled.md"))
        assert list((tmp_path / "output" / "drafts").rglob("cover_letter-filled.md"))
        assert "Filled copy created" in result.stdout

    def test_no_init_filled_flag_skips_filled_copies(
        self, fixture_vault: Path, sample_llm_output, tmp_path: Path
    ):
        result = self._invoke_match_and_draft(
            fixture_vault, sample_llm_output, tmp_path, extra_args=["--no-init-filled"]
        )
        assert result.exit_code == 0, result.stdout
        assert not list((tmp_path / "output" / "drafts").rglob("resume-filled.md"))
        assert not list((tmp_path / "output" / "drafts").rglob("cover_letter-filled.md"))

    def test_rerun_does_not_clobber_existing_filled_copy(
        self, fixture_vault: Path, sample_llm_output, tmp_path: Path
    ):
        self._invoke_match_and_draft(fixture_vault, sample_llm_output, tmp_path)
        filled_path = next((tmp_path / "output" / "drafts").rglob("resume-filled.md"))
        filled_path.write_text("# Hand-edited content, do not lose", encoding="utf-8")

        result = self._invoke_match_and_draft(fixture_vault, sample_llm_output, tmp_path)

        assert result.exit_code == 0, result.stdout
        assert filled_path.read_text() == "# Hand-edited content, do not lose"
        assert "already exists, left untouched" in result.stdout


class TestTrackCommands:
    def test_add_then_list(self, tmp_path: Path):
        tracker_path = tmp_path / "applications.json"

        add_result = runner.invoke(
            app,
            [
                "track", "add",
                "--company", "Acme",
                "--role", "Data Scientist",
                "--variant", "data-science",
                "--register", "formal-professional",
                "--tracker-path", str(tracker_path),
            ],
        )
        assert add_result.exit_code == 0, add_result.stdout

        list_result = runner.invoke(app, ["track", "list", "--tracker-path", str(tracker_path)])
        assert list_result.exit_code == 0
        assert "Acme" in list_result.stdout
        assert "Data Scientist" in list_result.stdout

    def test_add_then_update_then_show(self, tmp_path: Path):
        tracker_path = tmp_path / "applications.json"
        add_result = runner.invoke(
            app,
            [
                "track", "add",
                "--company", "Acme",
                "--role", "Data Scientist",
                "--variant", "data-science",
                "--register", "formal-professional",
                "--tracker-path", str(tracker_path),
            ],
        )
        record_id = add_result.stdout.split()[2]

        update_result = runner.invoke(
            app,
            [
                "track", "update", record_id,
                "--status", "applied",
                "--tracker-path", str(tracker_path),
            ],
        )
        assert update_result.exit_code == 0, update_result.stdout

        show_result = runner.invoke(
            app, ["track", "show", record_id, "--tracker-path", str(tracker_path)]
        )
        assert show_result.exit_code == 0
        assert '"status": "applied"' in show_result.stdout

    def test_update_missing_id_errors_cleanly(self, tmp_path: Path):
        tracker_path = tmp_path / "applications.json"
        result = runner.invoke(
            app,
            ["track", "update", "not-a-real-id", "--status", "applied", "--tracker-path", str(tracker_path)],
        )
        assert result.exit_code != 0

    def test_update_with_no_fields_errors_cleanly(self, tmp_path: Path):
        tracker_path = tmp_path / "applications.json"
        runner.invoke(
            app,
            [
                "track", "add",
                "--company", "Acme", "--role", "DS",
                "--variant", "data-science", "--register", "formal-professional",
                "--tracker-path", str(tracker_path),
            ],
        )
        result = runner.invoke(app, ["track", "update", "some-id", "--tracker-path", str(tracker_path)])
        assert result.exit_code != 0

    def test_list_empty_tracker(self, tmp_path: Path):
        tracker_path = tmp_path / "applications.json"
        result = runner.invoke(app, ["track", "list", "--tracker-path", str(tracker_path)])
        assert result.exit_code == 0
        assert "No matching applications." in result.stdout
