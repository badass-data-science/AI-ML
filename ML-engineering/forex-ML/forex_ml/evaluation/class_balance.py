"""Class balance reporting — a cheap check for silent regime drift.

The training target is triple-barrier labeling (see TimeSeriesSplitter/
forex_ml.data.triple_barrier): each row is labeled by whichever of a profit-take,
stop-loss, or max-holding-period barrier is hit first. Train's class balance isn't
guaranteed to be even the way percentile-threshold binning used to make it (there's
no train-quantile fitting step anymore), and val/test can drift further still if the
volatility/trend regime shifts between periods (routine for FX) — a market regime
where profit-take barriers get hit often can look completely different a year later.
Reporting the actual balance for all three splits, every run, surfaces that drift
directly instead of leaving it hidden inside a single accuracy number.
"""

from __future__ import annotations

import numpy as np


def class_balance(y: np.ndarray) -> dict[str, float]:
    """Fraction of rows in each outcome class, keyed by class index."""
    class_idx = np.argmax(y, axis=1)
    counts = np.bincount(class_idx, minlength=y.shape[1])
    return {f"class_{i}": float(count) / len(class_idx) for i, count in enumerate(counts)}
