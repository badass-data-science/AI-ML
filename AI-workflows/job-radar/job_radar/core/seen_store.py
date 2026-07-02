"""Flat-JSON-backed record of every posting job-radar has ever pulled.

Mirrors job-hunt-agent's core/tracker.py ApplicationStore: load-whole-file,
mutate, save-whole-file. This store is what turns "posted_at" into a
trustworthy staleness signal — an ATS's own dates can be wrong or absent, but
if job-radar has independently observed the same posting across N `pull` runs
spanning M days, that's first-party evidence a scraped date can't fake.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from job_radar.core.models import SeenRecord


class SeenPostingStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}", encoding="utf-8")

    def _read_all(self) -> dict[str, SeenRecord]:
        raw = self.path.read_text(encoding="utf-8").strip() or "{}"
        data = json.loads(raw)
        return {k: SeenRecord(**v) for k, v in data.items()}

    def _write_all(self, records: dict[str, SeenRecord]) -> None:
        data = {k: r.model_dump(mode="json") for k, r in records.items()}
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get(self, key: str) -> SeenRecord | None:
        return self._read_all().get(key)

    def record_seen(self, key: str, title: str, source_updated_at: datetime | None) -> SeenRecord:
        """Upsert a sighting for `key` — increments seen_count if known, else creates it."""
        records = self._read_all()
        now = datetime.now()
        existing = records.get(key)
        if existing is None:
            record = SeenRecord(
                key=key,
                first_seen_at=now,
                last_seen_at=now,
                seen_count=1,
                last_title=title,
                last_source_updated_at=source_updated_at,
            )
        else:
            record = existing.model_copy(
                update={
                    "last_seen_at": now,
                    "seen_count": existing.seen_count + 1,
                    "last_title": title,
                    "last_source_updated_at": source_updated_at,
                }
            )
        records[key] = record
        self._write_all(records)
        return record
