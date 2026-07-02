"""Tests for ats_clients.py — httpx is mocked via patch(), never hits the network.

PATCH TARGET: we patch "job_radar.core.ats_clients.httpx.AsyncClient.get", i.e.
the name as looked up from ats_clients.py's own `import httpx`, not the
`httpx` package globally — same discipline strategic-reports' test_ingestion.py
documents for patching feedparser.parse.
"""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from job_radar.core.ats_clients import fetch_ashby, fetch_greenhouse, fetch_lever
from job_radar.core.models import CompanyConfig


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _patched_get(payload):
    return patch(
        "job_radar.core.ats_clients.httpx.AsyncClient.get",
        new=AsyncMock(return_value=_FakeResponse(payload)),
    )


class TestFetchGreenhouse:
    async def test_parses_jobs_into_raw_postings(self):
        payload = {
            "jobs": [
                {
                    "id": 42,
                    "title": "Data Scientist",
                    "updated_at": "2026-06-01T00:00:00Z",
                    "absolute_url": "https://boards.greenhouse.io/acme/jobs/42",
                    "location": {"name": "Remote"},
                    "departments": [{"name": "Engineering"}],
                    "content": "<p>Great role</p>",
                }
            ]
        }
        with _patched_get(payload):
            postings = await fetch_greenhouse(CompanyConfig(name="Acme", ats="greenhouse", slug="acme"))
        assert len(postings) == 1
        p = postings[0]
        assert p.external_id == "42"
        assert p.title == "Data Scientist"
        assert p.location == "Remote"
        assert p.department == "Engineering"
        assert "Great role" in p.raw_text
        assert p.posted_at is None
        assert p.updated_at is not None

    async def test_html_entity_escaped_content_is_unescaped_before_conversion(self):
        """Regression test for a real bug found via manual smoke-testing against
        the live Figma Greenhouse board: `content` arrives HTML-entity-escaped
        (literal `&lt;div&gt;` text), not real markup. Without unescaping first,
        html_to_markdown finds no real tags to strip and raw-looking escaped
        markup leaks straight into raw_text.
        """
        payload = {
            "jobs": [
                {
                    "id": 42,
                    "title": "Data Scientist",
                    "updated_at": "2026-06-01T00:00:00Z",
                    "absolute_url": "https://boards.greenhouse.io/acme/jobs/42",
                    "location": {"name": "Remote"},
                    "departments": [{"name": "Engineering"}],
                    "content": "&lt;div class=&quot;content-intro&quot;&gt;&lt;p&gt;Great role&lt;/p&gt;&lt;/div&gt;",
                }
            ]
        }
        with _patched_get(payload):
            postings = await fetch_greenhouse(CompanyConfig(name="Acme", ats="greenhouse", slug="acme"))
        assert "Great role" in postings[0].raw_text
        assert "&lt;" not in postings[0].raw_text
        assert "<div" not in postings[0].raw_text

    async def test_network_failure_returns_empty_list(self):
        with patch(
            "job_radar.core.ats_clients.httpx.AsyncClient.get",
            new=AsyncMock(side_effect=httpx.ConnectError("boom")),
        ):
            postings = await fetch_greenhouse(CompanyConfig(name="Acme", ats="greenhouse", slug="acme"))
        assert postings == []


class TestFetchLever:
    async def test_parses_postings_list(self):
        payload = [
            {
                "id": "abc123",
                "text": "Senior Engineer",
                "createdAt": 1750000000000,
                "hostedUrl": "https://jobs.lever.co/acme/abc123",
                "categories": {"location": "NYC", "team": "Platform"},
                "descriptionPlain": "Build things.",
            }
        ]
        with _patched_get(payload):
            postings = await fetch_lever(CompanyConfig(name="Acme", ats="lever", slug="acme"))
        assert len(postings) == 1
        p = postings[0]
        assert p.external_id == "abc123"
        assert p.title == "Senior Engineer"
        assert p.location == "NYC"
        assert p.department == "Platform"
        assert p.raw_text == "Build things."
        assert p.posted_at is not None

    async def test_network_failure_returns_empty_list(self):
        with patch(
            "job_radar.core.ats_clients.httpx.AsyncClient.get",
            new=AsyncMock(side_effect=httpx.ConnectError("boom")),
        ):
            postings = await fetch_lever(CompanyConfig(name="Acme", ats="lever", slug="acme"))
        assert postings == []


class TestFetchAshby:
    async def test_parses_job_postings(self):
        payload = {
            "jobs": [
                {
                    "id": "xyz789",
                    "title": "ML Engineer",
                    "publishedAt": "2026-06-15T00:00:00Z",
                    "jobUrl": "https://jobs.ashbyhq.com/acme/xyz789",
                    "location": "SF",
                    "department": "AI",
                    "descriptionPlain": "Ship models.",
                }
            ]
        }
        with _patched_get(payload):
            postings = await fetch_ashby(CompanyConfig(name="Acme", ats="ashby", slug="acme"))
        assert len(postings) == 1
        p = postings[0]
        assert p.external_id == "xyz789"
        assert p.title == "ML Engineer"
        assert p.location == "SF"
        assert p.department == "AI"
        assert p.raw_text == "Ship models."
        assert p.posted_at is not None

    async def test_network_failure_returns_empty_list(self):
        with patch(
            "job_radar.core.ats_clients.httpx.AsyncClient.get",
            new=AsyncMock(side_effect=httpx.ConnectError("boom")),
        ):
            postings = await fetch_ashby(CompanyConfig(name="Acme", ats="ashby", slug="acme"))
        assert postings == []
