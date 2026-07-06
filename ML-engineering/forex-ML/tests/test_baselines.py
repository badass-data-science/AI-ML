from __future__ import annotations

import numpy as np

from forex_ml.evaluation.baselines import majority_class_baseline, persistence_baseline


def _one_hot(class_indices: list[int], n_classes: int = 3) -> np.ndarray:
    return np.eye(n_classes)[class_indices]


def test_majority_class_baseline_predicts_most_common_train_class():
    y_train = _one_hot([0, 0, 0, 1, 2])  # class 0 is most common
    y_eval = _one_hot([0, 1, 0, 2])  # 2 of 4 rows are actually class 0

    result = majority_class_baseline(y_train, y_eval)
    assert result["accuracy"] == 0.5


def test_persistence_baseline_perfect_when_class_never_changes():
    y_eval = _one_hot([0, 0, 0, 0])
    result = persistence_baseline(y_eval)
    assert result["accuracy"] == 1.0


def test_persistence_baseline_zero_when_class_always_alternates():
    y_eval = _one_hot([0, 1, 0, 1])
    result = persistence_baseline(y_eval)
    assert result["accuracy"] == 0.0


def test_persistence_baseline_scores_n_minus_one_predictions():
    """Row 0 has no previous period to persist from, so with n rows there are only
    n-1 scoreable predictions: pred[i] = actual[i-1] for i in 1..n-1."""
    y_eval = _one_hot([1, 0, 0, 0])
    # pred = eval[:-1] = [1, 0, 0], actual = eval[1:] = [0, 0, 0] -> 2 of 3 correct
    result = persistence_baseline(y_eval)
    assert result["accuracy"] == 2 / 3


def test_persistence_baseline_returns_nan_for_single_row():
    y_eval = _one_hot([0])
    result = persistence_baseline(y_eval)
    assert np.isnan(result["accuracy"])
