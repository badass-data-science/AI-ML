from pathlib import Path

import pytest

from job_radar.core.seen_store import SeenPostingStore


@pytest.fixture
def store(tmp_path: Path) -> SeenPostingStore:
    return SeenPostingStore(tmp_path / "seen_store.json")


class TestSeenPostingStore:
    def test_creates_empty_store_file(self, tmp_path: Path):
        store = SeenPostingStore(tmp_path / "seen_store.json")
        assert store.path.exists()
        assert store.get("greenhouse:acme:1") is None

    def test_first_sighting_creates_record_with_count_one(self, store: SeenPostingStore):
        record = store.record_seen("greenhouse:acme:1", "Data Scientist", None)
        assert record.seen_count == 1
        assert record.first_seen_at == record.last_seen_at
        assert record.last_title == "Data Scientist"

    def test_second_sighting_increments_count_and_preserves_first_seen(self, store: SeenPostingStore):
        first = store.record_seen("greenhouse:acme:1", "Data Scientist", None)
        second = store.record_seen("greenhouse:acme:1", "Senior Data Scientist", None)
        assert second.seen_count == 2
        assert second.first_seen_at == first.first_seen_at
        assert second.last_title == "Senior Data Scientist"

    def test_persists_across_instances(self, tmp_path: Path):
        path = tmp_path / "seen_store.json"
        store1 = SeenPostingStore(path)
        store1.record_seen("greenhouse:acme:1", "Data Scientist", None)
        store2 = SeenPostingStore(path)
        record = store2.get("greenhouse:acme:1")
        assert record is not None
        assert record.seen_count == 1

    def test_different_keys_tracked_independently(self, store: SeenPostingStore):
        store.record_seen("greenhouse:acme:1", "Data Scientist", None)
        store.record_seen("lever:acme:1", "Data Scientist", None)
        assert store.get("greenhouse:acme:1").seen_count == 1
        assert store.get("lever:acme:1").seen_count == 1
