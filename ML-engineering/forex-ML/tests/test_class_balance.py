from __future__ import annotations

import numpy as np
import pytest

from forex_ml.evaluation.class_balance import class_balance


def _one_hot(class_indices: list[int], n_classes: int = 3) -> np.ndarray:
    return np.eye(n_classes)[class_indices]


def test_even_split_reports_even_balance():
    y = _one_hot([0, 0, 1, 1, 2, 2])
    result = class_balance(y)
    assert result == {"class_0": 1 / 3, "class_1": 1 / 3, "class_2": 1 / 3}


def test_skewed_split_reports_skew():
    y = _one_hot([0, 0, 0, 0, 1, 2])  # regime-drifted: mostly class 0
    result = class_balance(y)
    assert result["class_0"] == 4 / 6
    assert result["class_1"] == 1 / 6
    assert result["class_2"] == 1 / 6


def test_missing_class_reports_zero_not_a_missing_key():
    y = _one_hot([0, 0, 1, 1])  # class 2 never occurs in this split
    result = class_balance(y)
    assert result["class_2"] == 0.0


def test_balances_sum_to_one():
    rng = np.random.default_rng(0)
    y = _one_hot(list(rng.integers(0, 3, size=100)))
    result = class_balance(y)
    assert sum(result.values()) == pytest.approx(1.0)
