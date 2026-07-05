"""Async fetchers for each supported ATS's public job-board JSON API.

Design mirrors strategic-reports' core/ingestion.py::_fetch_one_feed: each
fetch function owns its own try/except and returns [] on failure (logged via
structlog), so one bad company config — wrong slug, ATS outage, network
hiccup — never takes down the rest of a `pull` run. Callers (source.py) run
these concurrently via asyncio.gather without needing their own error
handling.

None of these APIs require authentication or scraping — they're the same
public JSON endpoints each ATS's own hosted job board renders from client-
side, so this is no more sensitive than visiting the board in a browser.

Greenhouse's job-list endpoint doesn't reliably return a creation date (only
`updated_at`), so `posted_at` is left None there — this is precisely why
seen_store.py's own first-seen tracking exists as the primary staleness
signal rather than trusting each ATS's dates.
"""

from __future__ import annotations

import html as html_lib
from datetime import datetime

import html_to_markdown
import httpx
import structlog

from job_radar.core.models import CompanyConfig, RawPosting

log = structlog.get_logger(__name__)

_TIMEOUT = httpx.Timeout(15.0)


def _parse_iso(value: str | None) -> datetime | None:
    """Parse an ATS timestamp to a naive datetime.

    Every datetime in this pipeline (seen_store's datetime.now(), Lever's
    epoch-millis conversion via fromtimestamp) is naive, so tzinfo is
    dropped here rather than kept — mixing aware and naive datetimes raises
    TypeError the moment ghost_scoring subtracts one from `now`. A few hours
    of UTC-vs-local skew is immaterial at the day-granularity thresholds
    ghost_scoring uses.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _html_to_text(html: str | None) -> str:
    """Convert an ATS description field to plain text.

    Greenhouse's `content` field comes HTML-entity-escaped within the JSON
    value (e.g. literal `&lt;div&gt;` text, not a real `<div>` tag) — found
    via manual smoke-testing against a real board, where raw markup like
    `<div class="content-intro">` was leaking straight into RawPosting.raw_text
    because html_to_markdown had no real tags to strip, just escaped-looking
    text. html.unescape() first turns those entities back into real markup so
    html_to_markdown has something to actually convert.
    """
    if not html:
        return ""
    unescaped = html_lib.unescape(html)
    try:
        return html_to_markdown.convert(unescaped).content.strip()
    except Exception as exc:
        log.warning("html_to_text_failed", error=str(exc))
        return unescaped.strip()


async def fetch_greenhouse(company: CompanyConfig) -> list[RawPosting]:
    url = f"https://boards-api.greenhouse.io/v1/boards/{company.slug}/jobs"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url, params={"content": "true"})
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        log.warning("ats_fetch_failed", ats="greenhouse", company=company.name, slug=company.slug, error=str(exc))
        return []

    postings: list[RawPosting] = []
    for job in data.get("jobs", []):
        location = (job.get("location") or {}).get("name")
        departments = job.get("departments") or []
        department = departments[0]["name"] if departments else None
        postings.append(
            RawPosting(
                external_id=str(job["id"]),
                company=company.name,
                ats="greenhouse",
                title=job.get("title", "").strip(),
                location=location,
                department=department,
                url=job.get("absolute_url", ""),
                raw_text=_html_to_text(job.get("content")),
                posted_at=None,
                updated_at=_parse_iso(job.get("updated_at")),
            )
        )
    log.debug("ats_fetched", ats="greenhouse", company=company.name, count=len(postings))
    return postings


async def fetch_lever(company: CompanyConfig) -> list[RawPosting]:
    url = f"https://api.lever.co/v0/postings/{company.slug}"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url, params={"mode": "json"})
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        log.warning("ats_fetch_failed", ats="lever", company=company.name, slug=company.slug, error=str(exc))
        return []

    postings: list[RawPosting] = []
    for job in data:
        categories = job.get("categories") or {}
        created_ms = job.get("createdAt")
        posted_at = datetime.fromtimestamp(created_ms / 1000) if created_ms else None
        raw_text = job.get("descriptionPlain") or _html_to_text(job.get("description"))
        postings.append(
            RawPosting(
                external_id=str(job["id"]),
                company=company.name,
                ats="lever",
                title=job.get("text", "").strip(),
                location=categories.get("location"),
                department=categories.get("team"),
                url=job.get("hostedUrl", ""),
                raw_text=raw_text.strip(),
                posted_at=posted_at,
                updated_at=posted_at,
            )
        )
    log.debug("ats_fetched", ats="lever", company=company.name, count=len(postings))
    return postings


async def fetch_ashby(company: CompanyConfig) -> list[RawPosting]:
    url = f"https://api.ashbyhq.com/posting-api/job-board/{company.slug}"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url, params={"includeCompensation": "false"})
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        log.warning("ats_fetch_failed", ats="ashby", company=company.name, slug=company.slug, error=str(exc))
        return []

    postings: list[RawPosting] = []
    for job in data.get("jobs", []):
        posted_at = _parse_iso(job.get("publishedAt"))
        raw_text = job.get("descriptionPlain") or _html_to_text(job.get("descriptionHtml"))
        postings.append(
            RawPosting(
                external_id=str(job["id"]),
                company=company.name,
                ats="ashby",
                title=job.get("title", "").strip(),
                location=job.get("location"),
                department=job.get("department"),
                url=job.get("jobUrl") or job.get("applyUrl", ""),
                raw_text=raw_text.strip(),
                posted_at=posted_at,
                updated_at=posted_at,
            )
        )
    log.debug("ats_fetched", ats="ashby", company=company.name, count=len(postings))
    return postings


FETCHERS = {
    "greenhouse": fetch_greenhouse,
    "lever": fetch_lever,
    "ashby": fetch_ashby,
}
