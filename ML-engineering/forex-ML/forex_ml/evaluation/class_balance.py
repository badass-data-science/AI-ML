"""Class balance reporting — a cheap check for silent regime drift.

The tertile class thresholds are computed once from TRAINING-period pd_lead
quantiles (correctly — see TimeSeriesSplitter), so by construction train's class
balance is close to even. Val/test are NOT guaranteed to be: if volatility regime
shifts between periods (routine for FX), a threshold calibrated on a calm training
period can make a later volatile test period look mostly "extreme class", or vice
versa. Reporting the actual balance for all three splits, every run, surfaces that
drift directly instead of leaving it hidden inside a single accuracy number.
"""

from __future__ import annotations

import numpy as np


def class_balance(y: np.ndarray) -> dict[str, float]:
    """Fraction of rows in each outcome class, keyed by class index."""
    class_idx = np.argmax(y, axis=1)
    counts = np.bincount(class_idx, minlength=y.shape[1])
    return {f"class_{i}": float(count) / len(class_idx) for i, count in enumerate(counts)}
