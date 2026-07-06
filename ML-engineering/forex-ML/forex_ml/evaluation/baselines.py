"""Baseline forecasts to contextualize the LSTM's own test metrics.

Neither baseline touches X or the model — both are computed purely from the
already time-ordered `y` (one-hot outcome) arrays. If the LSTM can't beat these, it
isn't learning anything useful from the engineered features; it's just an expensive
way to approximate a trivial rule.
"""

from __future__ import annotations

import numpy as np


def _accuracy(y_true_idx: np.ndarray, y_pred_idx: np.ndarray) -> float:
    return float(np.mean(y_true_idx == y_pred_idx))


def majority_class_baseline(y_train: np.ndarray, y_eval: np.ndarray) -> dict[str, float]:
    """Always predict whichever class was most common in the training set — the
    simplest possible baseline: a "model" that ignores the input entirely."""
    train_idx = np.argmax(y_train, axis=1)
    majority_class = int(np.bincount(train_idx).argmax())

    eval_idx = np.argmax(y_eval, axis=1)
    predictions = np.full_like(eval_idx, majority_class)
    return {"accuracy": _accuracy(eval_idx, predictions)}


def persistence_baseline(y_eval: np.ndarray) -> dict[str, float]:
    """Predict that this period's class is the same as the PREVIOUS period's actual
    class — the classic time-series "naive forecast" baseline, meaningful here
    specifically because FX volatility clusters (a big move tends to follow another
    big move). `y_eval` must already be in chronological order (true of every Splits
    array in this pipeline). The first row has no previous period to persist from, so
    it's excluded from scoring rather than padded with a guess.
    """
    eval_idx = np.argmax(y_eval, axis=1)
    if len(eval_idx) < 2:
        return {"accuracy": float("nan")}
    predictions = eval_idx[:-1]
    actuals = eval_idx[1:]
    return {"accuracy": _accuracy(actuals, predictions)}
