from __future__ import annotations

import numpy as np
import pytest

from forex_strategy.backtest import predicted_classes_to_positions


def test_predicted_classes_to_positions_maps_terciles_to_short_flat_long():
    pred_proba = np.array([
        [0.7, 0.2, 0.1],  # class 0 (lowest tercile) -> short
        [0.1, 0.8, 0.1],  # class 1 (middle) -> flat
        [0.1, 0.2, 0.7],  # class 2 (highest tercile) -> long
    ])
    positions = predicted_classes_to_positions(pred_proba)
    np.testing.assert_array_equal(positions, [-1, 0, 1])


def test_predicted_classes_to_positions_min_confidence_forces_flat():
    pred_proba = np.array([
        [0.4, 0.3, 0.3],  # winning class 0, but low confidence
        [0.9, 0.05, 0.05],  # winning class 0, high confidence
    ])
    positions = predicted_classes_to_positions(pred_proba, min_confidence=0.6)
    np.testing.assert_array_equal(positions, [0, -1])


def test_predicted_classes_to_positions_rejects_wrong_shape():
    with pytest.raises(ValueError, match="3"):
        predicted_classes_to_positions(np.array([[0.5, 0.5]]))
