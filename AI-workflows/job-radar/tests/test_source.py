import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from job_radar.core.models import CompanyConfig
from job_radar.core.seen_store import SeenPostingStore
from job_radar.core.source import load_companies, pull_postings
from tests.conftest import make_posting


class TestLoadCompanies:
    def test_loads_companies_from_json(self, tmp_path: Path):
        path = tmp_path / "companies.json"
        path.write_text(
            json.dumps(
                {
                    "companies": [
                        {"name": "Acme", "ats": "greenhouse", "slug": "acme"},
                        {"name": "Globex", "ats": "lever", "slug": "globex"},
                    ]
                }
            )
        )
        companies = load_companies(path)
        assert companies == [
            CompanyConfig(name="Acme", ats="greenhouse", slug="acme"),
            CompanyConfig(name="Globex", ats="lever", slug="globex"),
        ]

    def test_ignores_top_level_comment_key(self, tmp_path: Path):
        path = tmp_path / "companies.json"
        path.write_text(
            json.dumps({"_comment": "example only", "companies": [{"name": "Acme", "ats": "greenhouse", "slug": "acme"}]})
        )
        companies = load_companies(path)
        assert len(companies) == 1


class TestPullPostings:
    async def test_dispatches_to_correct_fetcher_and_scores(self, tmp_path: Path):
        acme = CompanyConfig(name="Acme", ats="greenhouse", slug="acme")
        globex = CompanyConfig(name="Globex", ats="lever", slug="globex")

        acme_posting = make_posting(external_id="1", company="Acme", ats="greenhouse")
        globex_posting = make_posting(external_id="2", company="Globex", ats="lever")

        fake_fetchers = {
            "greenhouse": AsyncMock(return_value=[acme_posting]),
            "lever": AsyncMock(return_value=[globex_posting]),
            "ashby": AsyncMock(return_value=[]),
        }

        seen_store = SeenPostingStore(tmp_path / "seen_store.json")

        with patch("job_radar.core.source.FETCHERS", fake_fetchers):
            scored = await pull_postings([acme, globex], seen_store)

        assert len(scored) == 2
        fake_fetchers["greenhouse"].assert_awaited_once_with(acme)
        fake_fetchers["lever"].assert_awaited_once_with(globex)
        fake_fetchers["ashby"].assert_not_awaited()

        assert all(sp.seen_count == 1 for sp in scored)
        assert seen_store.get("greenhouse:acme:1") is not None
        assert seen_store.get("lever:globex:2") is not None

    async def test_sorts_by_ghost_score_ascending(self, tmp_path: Path):
        acme = CompanyConfig(name="Acme", ats="greenhouse", slug="acme")
        risky = make_posting(
            external_id="1", posted_at=None, raw_text="always hiring, join our talent pool", title="Riskier"
        )
        safe = make_posting(external_id="2", title="Safer")

        fake_fetchers = {
            "greenhouse": AsyncMock(return_value=[risky, safe]),
            "lever": AsyncMock(return_value=[]),
            "ashby": AsyncMock(return_value=[]),
        }
        seen_store = SeenPostingStore(tmp_path / "seen_store.json")

        with patch("job_radar.core.source.FETCHERS", fake_fetchers):
            scored = await pull_postings([acme], seen_store)

        assert [sp.posting.title for sp in scored] == ["Safer", "Riskier"]

    async def test_repeated_pull_increments_seen_count(self, tmp_path: Path):
        acme = CompanyConfig(name="Acme", ats="greenhouse", slug="acme")
        posting = make_posting(external_id="1")
        fake_fetchers = {
            "greenhouse": AsyncMock(return_value=[posting]),
            "lever": AsyncMock(return_value=[]),
            "ashby": AsyncMock(return_value=[]),
        }
        seen_store = SeenPostingStore(tmp_path / "seen_store.json")

        with patch("job_radar.core.source.FETCHERS", fake_fetchers):
            await pull_postings([acme], seen_store)
            scored_second = await pull_postings([acme], seen_store)

        assert scored_second[0].seen_count == 2
