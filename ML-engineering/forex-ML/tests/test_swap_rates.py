from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from forex_ml.data.swap_rates import (
    _annual_rate_to_swap_cost_pct_per_night,
    fetch_current_swap_rates,
    resolve_swap_cost_pct_per_night,
)


def test_annual_rate_to_swap_cost_pct_per_night_negative_rate_becomes_a_positive_cost():
    # EUR/USD long_rate observed live: -0.0248 (annual decimal) -> a small real
    # per-night cost, positive per triple_barrier.py's "positive = charged" convention.
    result = _annual_rate_to_swap_cost_pct_per_night(-0.0248)
    assert result == pytest.approx(0.0248 * 100 / 365)
    assert result > 0


def test_annual_rate_to_swap_cost_pct_per_night_positive_rate_becomes_a_negative_cost():
    # EUR/USD short_rate observed live: +0.0046 -- a credit, so a NEGATIVE cost
    # (net gain), not a positive one.
    result = _annual_rate_to_swap_cost_pct_per_night(0.0046)
    assert result == pytest.approx(-0.0046 * 100 / 365)
    assert result < 0


def test_annual_rate_to_swap_cost_pct_per_night_zero_rate_is_zero_cost():
    assert _annual_rate_to_swap_cost_pct_per_night(0.0) == 0.0


def _mock_ifc_returning(df: pd.DataFrame):
    mock_ifc = MagicMock()
    mock_ifc.run_flux_query_on_forex_database_and_get_dataframe.return_value = df
    return patch("forex_ml.data.swap_rates._make_ifc", return_value=mock_ifc)


def test_fetch_current_swap_rates_parses_a_pivoted_row():
    df = pd.DataFrame({
        "unix_epoch_s": [1783487364],
        "long_rate": [-0.0248],
        "short_rate": [0.0046],
        "instrument": ["EUR/USD"],
    })
    with _mock_ifc_returning(df):
        result = fetch_current_swap_rates("EUR/USD")

    assert result is not None
    long_swap, short_swap = result
    assert long_swap == pytest.approx(0.0248 * 100 / 365)
    assert short_swap == pytest.approx(-0.0046 * 100 / 365)


def test_fetch_current_swap_rates_returns_none_on_empty_result():
    df = pd.DataFrame({"unix_epoch_s": [], "long_rate": [], "short_rate": [], "instrument": []})
    with _mock_ifc_returning(df):
        assert fetch_current_swap_rates("EUR/USD") is None


def test_fetch_current_swap_rates_returns_none_if_fields_missing():
    """A row came back (e.g. a stray point on the wrong measurement got through a
    misconfigured query) but doesn't actually have the fields this function needs --
    treated the same as "no data," not a crash."""
    df = pd.DataFrame({"unix_epoch_s": [123], "instrument": ["EUR/USD"]})
    with _mock_ifc_returning(df):
        assert fetch_current_swap_rates("EUR/USD") is None


def test_resolve_swap_cost_pct_per_night_uses_live_data_when_available():
    with patch("forex_ml.data.swap_rates.fetch_current_swap_rates", return_value=(0.007, -0.001)):
        result = resolve_swap_cost_pct_per_night("EUR/USD", fallback_long=0.0)
    assert result == (0.007, -0.001)


def test_resolve_swap_cost_pct_per_night_falls_back_when_no_snapshot_exists():
    with patch("forex_ml.data.swap_rates.fetch_current_swap_rates", return_value=None):
        result = resolve_swap_cost_pct_per_night("EUR/USD", fallback_long=0.02)
    assert result == (0.02, 0.02)  # fallback_short defaults to fallback_long


def test_resolve_swap_cost_pct_per_night_falls_back_on_fetch_exception():
    with patch("forex_ml.data.swap_rates.fetch_current_swap_rates", side_effect=RuntimeError("boom")):
        result = resolve_swap_cost_pct_per_night("EUR/USD", fallback_long=0.02, fallback_short=0.05)
    assert result == (0.02, 0.05)


def test_resolve_swap_cost_pct_per_night_distinct_fallback_long_and_short():
    with patch("forex_ml.data.swap_rates.fetch_current_swap_rates", return_value=None):
        result = resolve_swap_cost_pct_per_night("EUR/USD", fallback_long=0.01, fallback_short=0.03)
    assert result == (0.01, 0.03)
