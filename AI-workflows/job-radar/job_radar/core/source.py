"""Orchestration: load company config, fetch concurrently, score, record.

Mirrors strategic-reports' core/ingestion.py::fetch_topic_articles —
asyncio.gather over independent, already-error-isolated fetches, since ATS
fetching is I/O-bound (waiting on network), not CPU-bound.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import structlog

from job_radar.core.ats_clients import FETCHERS
from job_radar.core.ghost_scoring import score_posting
from job_radar.core.models import CompanyConfig, ScoredPosting
from job_radar.core.seen_store import SeenPostingStore

log = structlog.get_logger(__name__)


def load_companies(path: Path) -> list[CompanyConfig]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [CompanyConfig(**c) for c in raw["companies"]]


async def pull_postings(
    companies: list[CompanyConfig], seen_store: SeenPostingStore
) -> list[ScoredPosting]:
    log.info("pulling_postings", company_count=len(companies))

    results = await asyncio.gather(*[FETCHERS[c.ats](c) for c in companies])

    scored: list[ScoredPosting] = []
    for company, postings in zip(companies, results):
        for posting in postings:
            key = f"{posting.ats}:{company.slug}:{posting.external_id}"
            seen = seen_store.record_seen(key, posting.title, posting.updated_at)
            ghost = score_posting(posting, seen)
            scored.append(
                ScoredPosting(
                    posting=posting,
                    ghost=ghost,
                    first_seen_at=seen.first_seen_at,
                    last_seen_at=seen.last_seen_at,
                    seen_count=seen.seen_count,
                )
            )

    scored.sort(key=lambda sp: sp.ghost.score)
    log.info("pulled_postings", total=len(scored))
    return scored
