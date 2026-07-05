from unittest.mock import AsyncMock, patch

from job_radar.core.slug_discovery import discover_company, slugify_candidates
from tests.conftest import make_posting


class TestSlugifyCandidates:
    def test_generates_real_slug_for_acadia_pharmaceuticals(self):
        candidates = slugify_candidates("Acadia Pharmaceuticals")
        assert "acadiapharmaceuticals" in candidates
        assert "acadia-pharmaceuticals" in candidates
        assert "acadia" in candidates

    def test_generates_real_slug_for_scale_ai(self):
        candidates = slugify_candidates("Scale AI")
        assert "scaleai" in candidates
        assert "scale-ai" in candidates
        assert "scale" in candidates

    def test_generates_real_slug_for_10x_genomics(self):
        candidates = slugify_candidates("10x Genomics")
        assert "10xgenomics" in candidates

    def test_drops_common_corporate_suffixes_in_core_variant(self):
        candidates = slugify_candidates("The Widget Company Inc")
        assert "widget" in candidates
        assert "thewidgetcompanyinc" in candidates  # raw variant still included

    def test_single_word_company_name(self):
        candidates = slugify_candidates("Anthropic")
        assert candidates == ["anthropic"]

    def test_empty_or_symbols_only_name_returns_no_candidates(self):
        assert slugify_candidates("!!!") == []


class TestDiscoverCompany:
    async def test_confirms_only_live_boards(self):
        acadia_posting = make_posting(external_id="1", company="Acadia Pharmaceuticals")

        async def fake_greenhouse(company):
            return [acadia_posting] if company.slug == "acadiapharmaceuticals" else []

        with patch(
            "job_radar.core.slug_discovery.FETCHERS",
            {
                "greenhouse": AsyncMock(side_effect=fake_greenhouse),
                "lever": AsyncMock(return_value=[]),
                "ashby": AsyncMock(return_value=[]),
            },
        ):
            matches = await discover_company("Acadia Pharmaceuticals")

        assert len(matches) == 1
        assert matches[0].ats == "greenhouse"
        assert matches[0].slug == "acadiapharmaceuticals"
        assert matches[0].job_count == 1

    async def test_no_matches_returns_empty_list(self):
        with patch(
            "job_radar.core.slug_discovery.FETCHERS",
            {
                "greenhouse": AsyncMock(return_value=[]),
                "lever": AsyncMock(return_value=[]),
                "ashby": AsyncMock(return_value=[]),
            },
        ):
            matches = await discover_company("Totally Fictional Company Xyz")
        assert matches == []

    async def test_multiple_matches_sorted_by_job_count_descending(self):
        many = [make_posting(external_id=str(i)) for i in range(10)]
        few = [make_posting(external_id="1")]

        async def fake_greenhouse(company):
            return many if company.slug == "scaleai" else []

        async def fake_ashby(company):
            return few if company.slug == "scale" else []

        with patch(
            "job_radar.core.slug_discovery.FETCHERS",
            {
                "greenhouse": AsyncMock(side_effect=fake_greenhouse),
                "lever": AsyncMock(return_value=[]),
                "ashby": AsyncMock(side_effect=fake_ashby),
            },
        ):
            matches = await discover_company("Scale AI")

        assert len(matches) == 2
        assert matches[0].job_count == 10
        assert matches[0].slug == "scaleai"
        assert matches[1].job_count == 1
        assert matches[1].slug == "scale"
