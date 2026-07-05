from pathlib import Path

import pytest

from job_hunt_agent.core.models import ApplicationRecord
from job_hunt_agent.core.tracker import ApplicationStore


@pytest.fixture
def store(tmp_path: Path) -> ApplicationStore:
    return ApplicationStore(tmp_path / "tracker" / "applications.json")


def _make_record(**overrides) -> ApplicationRecord:
    kwargs = dict(
        company="Acme Corp",
        role="Data Scientist",
        resume_variant="data-science",
        cover_letter_register="formal-professional",
    )
    kwargs.update(overrides)
    return ApplicationRecord(**kwargs)


class TestApplicationStore:
    def test_creates_empty_store_file(self, tmp_path: Path):
        store = ApplicationStore(tmp_path / "applications.json")
        assert store.path.exists()
        assert store.list_all() == []

    def test_add_and_list(self, store: ApplicationStore):
        record = _make_record()
        store.add(record)
        all_records = store.list_all()
        assert len(all_records) == 1
        assert all_records[0].id == record.id
        assert all_records[0].company == "Acme Corp"

    def test_persists_across_instances(self, tmp_path: Path):
        path = tmp_path / "applications.json"
        store1 = ApplicationStore(path)
        store1.add(_make_record())
        store2 = ApplicationStore(path)
        assert len(store2.list_all()) == 1

    def test_get_by_id(self, store: ApplicationStore):
        record = _make_record()
        store.add(record)
        found = store.get(record.id)
        assert found is not None
        assert found.id == record.id

    def test_get_missing_id_returns_none(self, store: ApplicationStore):
        assert store.get("does-not-exist") is None

    def test_update_changes_fields_and_bumps_updated_at(self, store: ApplicationStore):
        record = _make_record()
        store.add(record)
        original_updated_at = record.updated_at

        updated = store.update(record.id, status="applied", notes="submitted online")
        assert updated.status == "applied"
        assert updated.notes == "submitted online"
        assert updated.updated_at >= original_updated_at

        # confirm the change actually persisted, not just returned in memory
        refetched = store.get(record.id)
        assert refetched.status == "applied"
        assert refetched.notes == "submitted online"

    def test_update_missing_id_raises(self, store: ApplicationStore):
        with pytest.raises(KeyError):
            store.update("does-not-exist", status="applied")

    def test_filter_by_status(self, store: ApplicationStore):
        r1 = _make_record(company="Acme")
        r2 = _make_record(company="Globex")
        store.add(r1)
        store.add(r2)
        store.update(r2.id, status="applied")

        drafted = store.filter(status="drafted")
        applied = store.filter(status="applied")
        assert [r.id for r in drafted] == [r1.id]
        assert [r.id for r in applied] == [r2.id]

    def test_filter_by_company_case_insensitive(self, store: ApplicationStore):
        r1 = _make_record(company="Acme Corp")
        store.add(r1)
        assert len(store.filter(company="acme corp")) == 1
        assert len(store.filter(company="ACME CORP")) == 1
        assert len(store.filter(company="Nonexistent")) == 0

    def test_default_status_is_drafted(self, store: ApplicationStore):
        record = _make_record()
        store.add(record)
        assert store.get(record.id).status == "drafted"
