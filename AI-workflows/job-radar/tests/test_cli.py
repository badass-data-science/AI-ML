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

    def test_paths_prints_txt_file_paths_not_table(self, tmp_path: Path):
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

        result = runner.invoke(app, ["list", "--home", str(tmp_path), "--paths"])
        assert result.exit_code == 0
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        assert len(lines) == 1
        assert lines[0].endswith(".txt")
        assert "GHOST" not in result.stdout
        txt_path = Path(lines[0])
        assert txt_path.is_file()
        assert txt_path.read_text() == posting.raw_text

    def test_paths_with_no_matches_prints_nothing(self, tmp_path: Path):
        result = runner.invoke(app, ["list", "--home", str(tmp_path), "--paths"])
        assert result.exit_code == 0
        assert result.stdout == ""

    def test_title_contains_narrows_by_role(self, tmp_path: Path):
        companies_path = _write_companies_config(tmp_path)
        scientist = make_posting(external_id="1", title="Senior Data Scientist")
        engineer = make_posting(external_id="2", title="Software Engineer")

        with patch(
            "job_radar.core.source.FETCHERS", {"greenhouse": AsyncMock(return_value=[scientist, engineer])}
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

        result = runner.invoke(app, ["list", "--home", str(tmp_path), "--title-contains", "data scientist"])
        assert "Senior Data Scientist" in result.stdout
        assert "Software Engineer" not in result.stdout

    def test_location_contains_narrows_by_location(self, tmp_path: Path):
        companies_path = _write_companies_config(tmp_path)
        remote = make_posting(external_id="1", title="Remote Role", location="Remote - US")
        onsite = make_posting(external_id="2", title="Onsite Role", location="San Francisco, CA")

        with patch(
            "job_radar.core.source.FETCHERS", {"greenhouse": AsyncMock(return_value=[remote, onsite])}
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

        result = runner.invoke(app, ["list", "--home", str(tmp_path), "--location-contains", "remote"])
        assert "Remote Role" in result.stdout
        assert "Onsite Role" not in result.stdout


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


class TestDiscoverSlugCmd:
    def test_reports_confirmed_matches_without_writing_by_default(self, tmp_path: Path):
        companies_path = _write_companies_config(tmp_path)
        posting = make_posting(external_id="1")

        async def fake_greenhouse(company):
            return [posting] if company.slug == "acadiapharmaceuticals" else []

        with patch(
            "job_radar.core.slug_discovery.FETCHERS",
            {
                "greenhouse": AsyncMock(side_effect=fake_greenhouse),
                "lever": AsyncMock(return_value=[]),
                "ashby": AsyncMock(return_value=[]),
            },
        ):
            result = runner.invoke(
                app, ["discover-slug", "Acadia Pharmaceuticals", "--companies-path", str(companies_path)]
            )

        assert result.exit_code == 0
        assert "greenhouse" in result.stdout
        assert "acadiapharmaceuticals" in result.stdout
        # not written since --add wasn't passed
        assert "acadiapharmaceuticals" not in companies_path.read_text()

    def test_no_matches_reports_and_exits_cleanly(self, tmp_path: Path):
        companies_path = _write_companies_config(tmp_path)
        with patch(
            "job_radar.core.slug_discovery.FETCHERS",
            {
                "greenhouse": AsyncMock(return_value=[]),
                "lever": AsyncMock(return_value=[]),
                "ashby": AsyncMock(return_value=[]),
            },
        ):
            result = runner.invoke(
                app, ["discover-slug", "Totally Fictional Co", "--companies-path", str(companies_path)]
            )
        assert result.exit_code == 0
        assert "No live ATS board found" in result.stdout

    def test_add_with_single_match_writes_to_config(self, tmp_path: Path):
        companies_path = _write_companies_config(tmp_path)
        posting = make_posting(external_id="1")

        async def fake_greenhouse(company):
            return [posting] if company.slug == "acadiapharmaceuticals" else []

        with patch(
            "job_radar.core.slug_discovery.FETCHERS",
            {
                "greenhouse": AsyncMock(side_effect=fake_greenhouse),
                "lever": AsyncMock(return_value=[]),
                "ashby": AsyncMock(return_value=[]),
            },
        ):
            result = runner.invoke(
                app,
                ["discover-slug", "Acadia Pharmaceuticals", "--add", "--companies-path", str(companies_path)],
            )

        assert result.exit_code == 0
        assert "Added Acadia Pharmaceuticals" in result.stdout
        written = json.loads(companies_path.read_text())
        slugs = [(c["ats"], c["slug"]) for c in written["companies"]]
        assert ("greenhouse", "acadiapharmaceuticals") in slugs
        # original placeholder entry preserved, not clobbered
        assert ("greenhouse", "acme") in slugs

    def test_add_with_multiple_matches_refuses_without_pick(self, tmp_path: Path):
        companies_path = _write_companies_config(tmp_path)
        posting = make_posting(external_id="1")

        with patch(
            "job_radar.core.slug_discovery.FETCHERS",
            {
                "greenhouse": AsyncMock(return_value=[posting]),
                "lever": AsyncMock(return_value=[]),
                "ashby": AsyncMock(side_effect=lambda company: [posting] if company.slug == "acme" else []),
            },
        ):
            result = runner.invoke(
                app, ["discover-slug", "Acme", "--add", "--companies-path", str(companies_path)]
            )

        assert result.exit_code == 1
        assert "Multiple matches found" in result.stderr
        written = json.loads(companies_path.read_text())
        assert len(written["companies"]) == 1  # unchanged

    def test_add_with_pick_disambiguates(self, tmp_path: Path):
        companies_path = _write_companies_config(tmp_path)
        posting = make_posting(external_id="1")

        with patch(
            "job_radar.core.slug_discovery.FETCHERS",
            {
                "greenhouse": AsyncMock(return_value=[posting]),
                "lever": AsyncMock(return_value=[]),
                "ashby": AsyncMock(side_effect=lambda company: [posting] if company.slug == "acme" else []),
            },
        ):
            result = runner.invoke(
                app,
                [
                    "discover-slug", "Acme", "--add", "--pick", "ashby:acme",
                    "--companies-path", str(companies_path),
                ],
            )

        assert result.exit_code == 0
        written = json.loads(companies_path.read_text())
        slugs = [(c["ats"], c["slug"]) for c in written["companies"]]
        # the picked ashby:acme match was added as a new entry, alongside the
        # pre-existing greenhouse:acme placeholder — --pick doesn't touch it
        assert slugs.count(("ashby", "acme")) == 1
        assert slugs.count(("greenhouse", "acme")) == 1

    def test_add_is_idempotent_on_rerun(self, tmp_path: Path):
        companies_path = _write_companies_config(tmp_path)
        posting = make_posting(external_id="1")

        async def fake_greenhouse(company):
            return [posting] if company.slug == "acadiapharmaceuticals" else []

        with patch(
            "job_radar.core.slug_discovery.FETCHERS",
            {
                "greenhouse": AsyncMock(side_effect=fake_greenhouse),
                "lever": AsyncMock(return_value=[]),
                "ashby": AsyncMock(return_value=[]),
            },
        ):
            runner.invoke(
                app,
                ["discover-slug", "Acadia Pharmaceuticals", "--add", "--companies-path", str(companies_path)],
            )
            result = runner.invoke(
                app,
                ["discover-slug", "Acadia Pharmaceuticals", "--add", "--companies-path", str(companies_path)],
            )

        assert result.exit_code == 0
        assert "already in" in result.stdout
        written = json.loads(companies_path.read_text())
        slugs = [(c["ats"], c["slug"]) for c in written["companies"]]
        assert slugs.count(("greenhouse", "acadiapharmaceuticals")) == 1
