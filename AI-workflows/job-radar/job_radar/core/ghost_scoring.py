"""Deterministic ghost-job risk scoring — no LLM call.

Every signal here is structured (a date, a repeat count, a regex match), so
an LLM call would add cost and latency without improving on plain rules.
Score is a weighted sum of independent signals, clamped to [0, 1]; every
triggered signal appends a human-readable reason. Nothing here silently
excludes a posting — that's cli.py's opt-in `--max-ghost-score`, not this
module's job. Mirrors job-hunt-agent's guardrails.py philosophy: surface,
don't silently act.

Signal weights are deliberately coarse (tiers, not a fitted model) — there's
no labeled ghost-job dataset to fit against, and coarse, explainable
thresholds are easier for Emily to trust and tune by hand than an opaque
score.
"""

from __future__ import annotations

import re
from datetime import datetime

from job_radar.core.models import GhostSignal, RawPosting, SeenRecord

_EVERGREEN_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"always[\s-]hiring",
        r"talent (community|pool|network)",
        r"general application",
        r"future opportunit(y|ies)",
        r"evergreen",
        r"we'?re always looking",
    ]
]

_SHORT_DESCRIPTION_CHARS = 200
_AGE_WARN_DAYS = 45
_AGE_HIGH_DAYS = 90
_TRACKED_WARN_DAYS = 30
_TRACKED_HIGH_DAYS = 60


def score_posting(posting: RawPosting, seen: SeenRecord | None, now: datetime | None = None) -> GhostSignal:
    now = now or datetime.now()
    score = 0.0
    reasons: list[str] = []

    if posting.posted_at is not None:
        age_days = (now - posting.posted_at).days
        if age_days > _AGE_HIGH_DAYS:
            score += 0.35
            reasons.append(f"posted {age_days} days ago (>{_AGE_HIGH_DAYS}d)")
        elif age_days > _AGE_WARN_DAYS:
            score += 0.15
            reasons.append(f"posted {age_days} days ago (>{_AGE_WARN_DAYS}d)")

    if seen is not None:
        tracked_days = (now - seen.first_seen_at).days
        if tracked_days > _TRACKED_HIGH_DAYS:
            score += 0.35
            reasons.append(
                f"job-radar has tracked this posting for {tracked_days} days "
                f"across {seen.seen_count} pulls, still listed"
            )
        elif tracked_days > _TRACKED_WARN_DAYS and seen.seen_count >= 2:
            score += 0.15
            reasons.append(
                f"job-radar has tracked this posting for {tracked_days} days "
                f"across {seen.seen_count} pulls, still listed"
            )

    haystack = f"{posting.title}\n{posting.raw_text}"
    matched_phrases = sorted({m.group(0) for p in _EVERGREEN_PATTERNS if (m := p.search(haystack))})
    if matched_phrases:
        score += 0.25
        reasons.append(f"evergreen-hiring language detected: {', '.join(matched_phrases)}")

    text_len = len(posting.raw_text.strip())
    if text_len < _SHORT_DESCRIPTION_CHARS:
        score += 0.10
        reasons.append(f"description is unusually short ({text_len} chars) — may be a placeholder listing")

    return GhostSignal(score=min(score, 1.0), reasons=reasons)
