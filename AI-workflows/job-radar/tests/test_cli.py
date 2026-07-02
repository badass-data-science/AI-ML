import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from job_radar.cli import app
from tests.conftest import make_posting

runner = CliRunner()


def _write_companies_config(path: Path) -> Path:
    companies_path = path / "companies.json"
    companies_path.write_text(
        json.dumps({"companies": [{"name": "Acme", "ats": "greenhouse", "slug": "acme"}]})
    )
    return companies_path


class TestPullCmd:
    def test_writes_output_files_and_prints_summary(self, tmp_path: Path):
        companies_path = _write_companies_config(tmp_path)
        posting = make_posting(external_id="1", company="Acme", title="Data Scientist")

        with patch("job_radar.core.source.FETCHERS", {"greenhouse": AsyncMock(return_value=[posting])}):
            result = runner.invoke(
                app,
                [
                    "pull",
                    "--companies-path", str(companies_path),
                    "--seen-store-path", str(tmp_path / "seen_store.json"),
                    "--home", str(tmp_path),
                ],
            )

        assert result.exit_code == 0, result.stdout
        assert "Pulled 1 postings from 1 companies" in result.stdout
        assert "Data Scientist" in result.stdout

        json_files = list((tmp_path / "output" / "postings").rglob("*.json"))
        txt_files = list((tmp_path / "output" / "postings").rglob("*.txt"))
        assert len(json_files) == 1
        assert len(txt_files) == 1
        assert txt_files[0].read_text() == posting.raw_text


class TestListCmd:
    def test_lists_previously_pulled_postings(self, tmp_path: Path):
        companies_path = _write_companies_config(tmp_path)
        posting = make_posting(external_id="1", company="Acme", title="Data Scientist")

        with patch("job_radar.core.source.FETCHERS", {"greenhouse": AsyncMock(return_value=[posting])}):
            runner.invoke(
                app,
                [
                    "pull",
                    "--companies-path", str(companies_path),
                    "--seen-store-path", str(tmp_path / "seen_store.json"),
                    "--home", str(tmp_path),
                ],
            )

        result = runner.invoke(app, ["list", "--home", str(tmp_path)])
        assert result.exit_code == 0
        assert "Data Scientist" in result.stdout

    def test_max_ghost_score_hard_filters(self, tmp_path: Path):
        companies_path = _write_companies_config(tmp_path)
        risky = make_posting(
            external_id="1", posted_at=None, raw_text="always hiring, talent pool", title="Risky Role"
        )
        safe = make_posting(external_id="2", title="Safe Role")

        with patch(
            "job_radar.core.source.FETCHERS", {"greenhouse": AsyncMock(return_value=[risky, safe])}
        ):
            runner.invoke(
                app,
                [
                    "pull",
                    "--companies-path", str(companies_path),
                    "--seen-store-path", str(tmp_path / "seen_store.json"),
                    "--home", str(tmp_path),
                ],
            )

        result = runner.invoke(app, ["list", "--home", str(tmp_path), "--max-ghost-score", "0.1"])
        assert "Safe Role" in result.stdout
        assert "Risky Role" not in result.stdout

    def test_no_postings_dir_prints_message(self, tmp_path: Path):
        result = runner.invoke(app, ["list", "--home", str(tmp_path)])
        assert result.exit_code == 0
        assert "run `pull` first" in result.stdout


class TestShowCmd:
    def test_shows_full_detail(self, tmp_path: Path):
        companies_path = _write_companies_config(tmp_path)
        posting = make_posting(external_id="1", company="Acme", title="Data Scientist")

        with patch("job_radar.core.source.FETCHERS", {"greenhouse": AsyncMock(return_value=[posting])}):
            runner.invoke(
                app,
                [
                    "pull",
                    "--companies-path", str(companies_path),
                    "--seen-store-path", str(tmp_path / "seen_store.json"),
                    "--home", str(tmp_path),
                ],
            )

        json_file = next((tmp_path / "output" / "postings").rglob("*.json"))
        result = runner.invoke(app, ["show", str(json_file)])
        assert result.exit_code == 0
        assert "Data Scientist" in result.stdout
        assert "reasons" in result.stdout

    def test_missing_file_errors(self, tmp_path: Path):
        result = runner.invoke(app, ["show", str(tmp_path / "does-not-exist.json")])
        assert result.exit_code == 1
