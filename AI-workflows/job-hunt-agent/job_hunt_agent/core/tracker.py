"""Flat-JSON-backed application tracker.

Consistent with strategic-reports' own precedent for growing, date-queryable
history (bullet_history.json, urgency_history.json are flat JSON, not a DB).
The ApplicationStore interface is the only thing calling code touches, so a
future SQLite swap — if history ever genuinely outgrows this — is a localized
change, not a rewrite. This store never touches vault-Resume.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from job_hunt_agent.core.models import ApplicationRecord


class ApplicationStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def _read_all(self) -> list[ApplicationRecord]:
        raw = self.path.read_text(encoding="utf-8").strip() or "[]"
        return [ApplicationRecord(**d) for d in json.loads(raw)]

    def _write_all(self, records: list[ApplicationRecord]) -> None:
        data = [r.model_dump(mode="json") for r in records]
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def list_all(self) -> list[ApplicationRecord]:
        return self._read_all()

    def add(self, record: ApplicationRecord) -> ApplicationRecord:
        records = self._read_all()
        records.append(record)
        self._write_all(records)
        return record

    def get(self, record_id: str) -> ApplicationRecord | None:
        for r in self._read_all():
            if r.id == record_id:
                return r
        return None

    def update(self, record_id: str, **fields) -> ApplicationRecord:
        records = self._read_all()
        for i, r in enumerate(records):
            if r.id == record_id:
                updated = r.model_copy(update={**fields, "updated_at": datetime.now()})
                records[i] = updated
                self._write_all(records)
                return updated
        raise KeyError(f"no application record with id {record_id!r}")

    def filter(
        self, status: str | None = None, company: str | None = None
    ) -> list[ApplicationRecord]:
        records = self._read_all()
        if status is not None:
            records = [r for r in records if r.status == status]
        if company is not None:
            records = [r for r in records if r.company.lower() == company.lower()]
        return records
