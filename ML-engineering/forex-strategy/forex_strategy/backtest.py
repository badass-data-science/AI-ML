"""Turns a model's per-row class predictions into a trade decision (`positions`),
for handoff to the trade-simulator package's model-agnostic `simulate_trades`.

Class semantics match forex_ml.data.splitting.TimeSeriesSplitter._label_to_one_hot's
triple-barrier label mapping: class 0 = short's own profit-take independently
fired (label -1), class 1 = neither side's profit-take fired / flat (label 0),
class 2 = long's profit-take fired (label +1) -- so highest class = long signal,
lowest class = short signal, exactly the short/flat/long convention
`predicted_classes_to_positions` below assumes. This is deliberately forex-ML-
specific glue, not part of the trade-simulator package (github.com/badass-data-
science/forex-trade-simulation-inator) -- that package only ever sees
`positions`, never a model's raw prediction shape, precisely so it isn't tied
to this one labeling convention.
"""

from __future__ import annotations

import numpy as np


def predicted_classes_to_positions(pred_proba: np.ndarray, min_confidence: float = 0.0) -> np.ndarray:
    """+1 (long) for the highest-tercile class, -1 (short) for the lowest-tercile
    class, 0 (flat) for the middle class. `min_confidence` is an optional hurdle on
    the winning class's own probability -- rows below it are forced flat, since a
    low-confidence call is exactly the kind of trade least likely to clear costs."""
    if pred_proba.ndim != 2 or pred_proba.shape[1] != 3:
        raise ValueError(f"Expected a (n_rows, 3) array of class probabilities, got shape {pred_proba.shape}")

    pred_idx = np.argmax(pred_proba, axis=1)
    confidence = np.max(pred_proba, axis=1)
    positions = np.where(pred_idx == 2, 1, np.where(pred_idx == 0, -1, 0))
    return np.where(confidence >= min_confidence, positions, 0)
