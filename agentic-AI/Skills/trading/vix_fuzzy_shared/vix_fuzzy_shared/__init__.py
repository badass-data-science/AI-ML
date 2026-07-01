"""Shared VIX fuzzy-set constants and classification helpers.

Used by both vix-fuzzy-mcp-skill and risk-desk-mcp-skill so that VIX
fuzzy-set membership is computed identically everywhere in the trading
skill suite, instead of each skill keeping its own copy of the constants
and interpolation logic.
"""
from __future__ import annotations

import math

from python_tools_and_shortcuts.ai.fuzzylogic.FuzzyInterpolator import FuzzyInterpolator
from python_tools_and_shortcuts.econometrics.ticker_prices import get_most_recent_ticker_close_value

VIX_SET_NAMES = ['very low', 'low', 'medium low', 'medium', 'medium high', 'high', 'very high']

VIX_MEMBERSHIP_RANGES = {
    'very low':    [9.140000343322754,  12.869999885559082],
    'low':         [9.140000343322754,  12.869999885559082, 15.0600004196167],
    'medium low':  [12.869999885559082, 15.0600004196167,   17.450000762939453],
    'medium':      [15.0600004196167,   17.450000762939453, 20.649999618530273],
    'medium high': [17.450000762939453, 20.649999618530273, 25.110000610351562],
    'high':        [20.649999618530273, 25.110000610351562, 82.69000244140625],
    'very high':   [25.110000610351562, 82.69000244140625],
}

_interpolator = FuzzyInterpolator(VIX_SET_NAMES, VIX_MEMBERSHIP_RANGES)


def get_most_recent_vix() -> float:
    """Fetch and validate the most recent VIX close value."""
    vix = get_most_recent_ticker_close_value('^VIX')

    if not math.isfinite(vix):
        raise RuntimeError(f"Invalid VIX value returned: {vix!r}.")
    if vix < 0.0:
        raise RuntimeError(f"Unexpected negative VIX value returned: {vix!r}.")

    return vix


def interpolate_vix_membership(vix: float) -> dict:
    """Return the fuzzy interpolation result for a VIX value.

    The result has 'fuzzy set membership' (degree per VIX_SET_NAMES entry)
    and 'value range' keys.
    """
    return _interpolator.interpolate_membership(vix)
