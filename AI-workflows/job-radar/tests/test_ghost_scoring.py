from datetime import datetime, timedelta

from job_radar.core.ghost_scoring import score_posting
from job_radar.core.models import SeenRecord
from tests.conftest import make_posting


def _seen(first_seen_at: datetime, seen_count: int = 1, now: datetime | None = None) -> SeenRecord:
    now = now or datetime.now()
    return SeenRecord(
        key="greenhouse:acme:1",
        first_seen_at=first_seen_at,
        last_seen_at=now,
        seen_count=seen_count,
        last_title="Data Scientist",
    )


class TestAgeSignal:
    def test_fresh_posting_no_age_penalty(self):
        posting = make_posting(posted_at=datetime.now() - timedelta(days=2))
        result = score_posting(posting, seen=None)
        assert not any("posted" in r for r in result.reasons)

    def test_moderately_old_posting_gets_warn_tier(self):
        posting = make_posting(posted_at=datetime.now() - timedelta(days=60))
        result = score_posting(posting, seen=None)
        assert result.score >= 0.15
        assert any("60 days ago" in r for r in result.reasons)

    def test_very_old_posting_gets_high_tier(self):
        posting = make_posting(posted_at=datetime.now() - timedelta(days=120))
        result = score_posting(posting, seen=None)
        assert result.score >= 0.35
        assert any("120 days ago" in r for r in result.reasons)

    def test_missing_posted_at_skips_age_signal(self):
        posting = make_posting(posted_at=None)
        result = score_posting(posting, seen=None)
        assert not any("posted" in r for r in result.reasons)


class TestTrackedHistorySignal:
    def test_no_seen_record_skips_signal(self):
        posting = make_posting()
        result = score_posting(posting, seen=None)
        assert not any("tracked" in r for r in result.reasons)

    def test_long_tracked_history_triggers_high_tier(self):
        posting = make_posting()
        seen = _seen(first_seen_at=datetime.now() - timedelta(days=90), seen_count=5)
        result = score_posting(posting, seen=seen)
        assert result.score >= 0.35
        assert any("tracked" in r for r in result.reasons)

    def test_moderate_tracked_history_needs_repeat_sightings(self):
        posting = make_posting()
        seen_once = _seen(first_seen_at=datetime.now() - timedelta(days=40), seen_count=1)
        result_once = score_posting(posting, seen=seen_once)
        assert not any("tracked" in r for r in result_once.reasons)

        seen_twice = _seen(first_seen_at=datetime.now() - timedelta(days=40), seen_count=2)
        result_twice = score_posting(posting, seen=seen_twice)
        assert any("tracked" in r for r in result_twice.reasons)


class TestEvergreenLanguageSignal:
    def test_evergreen_phrase_in_title_detected(self):
        posting = make_posting(title="Always Hiring: Software Engineers", posted_at=None)
        result = score_posting(posting, seen=None)
        assert any("evergreen-hiring language" in r for r in result.reasons)

    def test_evergreen_phrase_in_body_detected(self):
        posting = make_posting(raw_text="Join our talent community for future opportunities. " * 5, posted_at=None)
        result = score_posting(posting, seen=None)
        assert any("evergreen-hiring language" in r for r in result.reasons)

    def test_ordinary_posting_no_evergreen_signal(self):
        posting = make_posting(posted_at=None)
        result = score_posting(posting, seen=None)
        assert not any("evergreen-hiring language" in r for r in result.reasons)


class TestShortDescriptionSignal:
    def test_short_description_flagged(self):
        posting = make_posting(raw_text="Short description.", posted_at=None)
        result = score_posting(posting, seen=None)
        assert any("unusually short" in r for r in result.reasons)

    def test_long_description_not_flagged(self):
        posting = make_posting(raw_text="A" * 500, posted_at=None)
        result = score_posting(posting, seen=None)
        assert not any("unusually short" in r for r in result.reasons)


class TestScoreClamping:
    def test_score_never_exceeds_one(self):
        posting = make_posting(
            posted_at=datetime.now() - timedelta(days=200),
            raw_text="always hiring, join our talent pool for future opportunities",
        )
        seen = _seen(first_seen_at=datetime.now() - timedelta(days=200), seen_count=10)
        result = score_posting(posting, seen=seen)
        assert result.score <= 1.0

    def test_clean_recent_posting_scores_near_zero(self):
        posting = make_posting(posted_at=datetime.now() - timedelta(days=1))
        result = score_posting(posting, seen=None)
        assert result.score == 0.0
        assert result.reasons == []
