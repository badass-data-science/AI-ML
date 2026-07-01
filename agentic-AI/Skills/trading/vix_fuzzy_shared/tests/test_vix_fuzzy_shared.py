"""Tests for vix-fuzzy-shared."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from vix_fuzzy_shared import (
    VIX_SET_NAMES,
    get_most_recent_vix,
    interpolate_vix_membership,
)


class TestGetMostRecentVix:
    def test_returns_value(self):
        with patch("vix_fuzzy_shared.get_most_recent_ticker_close_value", return_value=15.0):
            assert get_most_recent_vix() == 15.0

    def test_rejects_negative(self):
        with patch("vix_fuzzy_shared.get_most_recent_ticker_close_value", return_value=-1.0):
            with pytest.raises(RuntimeError):
                get_most_recent_vix()

    def test_rejects_non_finite(self):
        with patch("vix_fuzzy_shared.get_most_recent_ticker_close_value", return_value=float("nan")):
            with pytest.raises(RuntimeError):
                get_most_recent_vix()


class TestInterpolateVixMembership:
    def test_has_expected_keys(self):
        result = interpolate_vix_membership(15.0)
        assert set(result.keys()) == {"fuzzy set membership", "value range"}

    def test_membership_covers_all_sets(self):
        result = interpolate_vix_membership(15.0)
        assert set(result["fuzzy set membership"].keys()) == set(VIX_SET_NAMES)

    def test_clamps_out_of_range_value(self):
        # Values far outside the domain should not raise; they get clamped.
        result = interpolate_vix_membership(1000.0)
        assert result["fuzzy set membership"]["very high"] == pytest.approx(1.0)
