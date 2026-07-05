import pytest
from pydantic import ValidationError

from job_radar.core.models import CompanyConfig, GhostSignal
from tests.conftest import make_posting


def test_company_config_rejects_unknown_ats():
    with pytest.raises(ValidationError):
        CompanyConfig(name="Acme", ats="workday", slug="acme")


def test_ghost_signal_clamps_score_range():
    with pytest.raises(ValidationError):
        GhostSignal(score=1.5)
    with pytest.raises(ValidationError):
        GhostSignal(score=-0.1)


def test_raw_posting_defaults():
    posting = make_posting()
    assert posting.company == "Acme Corp"
    assert posting.ats == "greenhouse"
