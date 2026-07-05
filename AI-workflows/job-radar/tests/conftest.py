from datetime import datetime, timedelta

import pytest

from job_radar.core.models import CompanyConfig, RawPosting


@pytest.fixture
def company() -> CompanyConfig:
    return CompanyConfig(name="Acme Corp", ats="greenhouse", slug="acme")


def make_posting(**overrides) -> RawPosting:
    kwargs = dict(
        external_id="123",
        company="Acme Corp",
        ats="greenhouse",
        title="Data Scientist",
        location="Remote",
        department="Engineering",
        url="https://example.com/jobs/123",
        raw_text="A" * 500 + " We are looking for a data scientist with strong Python skills.",
        posted_at=datetime.now() - timedelta(days=5),
        updated_at=datetime.now() - timedelta(days=5),
    )
    kwargs.update(overrides)
    return RawPosting(**kwargs)
