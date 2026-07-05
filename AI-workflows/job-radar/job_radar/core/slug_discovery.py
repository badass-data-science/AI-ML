"""Guess-and-verify ATS slug discovery for a company name.

There is no cross-company "search by name" endpoint on any of the three ATS
platforms — only per-slug job lists (see ats_clients.py's module docstring
and README.md's "Finding companies to add" section). What this module
automates instead is the manual process already used to seed
config/companies.json by hand: generate a small set of plausible slug
spellings from a human-readable company name, then check each one against
every platform's real API and keep only the ones that come back with actual
live postings.

This never scrapes anything — every request here is the identical public
JSON API call ats_clients.py already makes for a real `pull`, just probing
speculative slugs instead of configured ones. Reuses FETCHERS directly so
there's exactly one place that knows how to talk to each platform.
"""

from __future__ import annotations

import asyncio
import re

import structlog

from job_radar.core.ats_clients import FETCHERS
from job_radar.core.models import ATSPlatform, CompanyConfig, DiscoveredSlug

log = structlog.get_logger(__name__)

_ATS_PLATFORMS: tuple[ATSPlatform, ...] = ("greenhouse", "lever", "ashby")

_CORPORATE_SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "llc",
    "ltd",
    "limited",
    "company",
    "the",
    "labs",
    "laboratories",
    "group",
    "holdings",
}


def slugify_candidates(company_name: str) -> list[str]:
    """Generate plausible ATS slug spellings for a human company name.

    e.g. "Acadia Pharmaceuticals" -> ["acadia", "acadia-pharmaceuticals",
    "acadiapharmaceuticals"] — which is exactly the real Greenhouse slug
    (verified by hand against the live board before this module existed).
    """
    words = re.findall(r"[a-z0-9]+", company_name.lower())
    core_words = [w for w in words if w not in _CORPORATE_SUFFIXES] or words

    candidates: set[str] = set()
    for word_list in (words, core_words):
        if not word_list:
            continue
        candidates.add("".join(word_list))
        candidates.add("-".join(word_list))
        if len(word_list) > 1:
            candidates.add(word_list[0])

    return sorted(candidates)


async def _probe(ats: ATSPlatform, slug: str) -> DiscoveredSlug | None:
    postings = await FETCHERS[ats](CompanyConfig(name=slug, ats=ats, slug=slug))
    if not postings:
        return None
    return DiscoveredSlug(ats=ats, slug=slug, job_count=len(postings))


async def discover_company(company_name: str) -> list[DiscoveredSlug]:
    """Probe every (candidate slug x platform) pair concurrently.

    Returns confirmed live matches only, sorted by job_count descending —
    a real board with real postings ranks above a coincidental slug
    collision that happens to return one or two jobs.
    """
    candidates = slugify_candidates(company_name)
    log.info("discovering_slugs", company=company_name, candidate_count=len(candidates))

    results = await asyncio.gather(
        *[_probe(ats, slug) for slug in candidates for ats in _ATS_PLATFORMS]
    )
    matches = [r for r in results if r is not None]
    matches.sort(key=lambda m: m.job_count, reverse=True)
    return matches
