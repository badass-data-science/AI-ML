from __future__ import annotations

import numpy as np
import pytest

from forex_strategy.backtest import predicted_classes_to_positions, simulate_trades


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


def test_simulate_trades_computes_net_pnl_and_only_charges_cost_on_trades():
    positions = np.array([1, -1, 0, 1])
    pd_lead_pct = np.array([0.10, -0.10, 0.50, -0.05])
    spread = np.full(4, 0.0002)
    price = np.full(4, 1.10)

    result = simulate_trades(positions, pd_lead_pct, spread, price)

    cost_per_trade_pct = 100.0 * 0.0002 / 1.10
    # row 0: long, +0.10% move -> gross = +0.10%
    # row 1: short, price fell -0.10% -> gross = (-1)*(-0.10) = +0.10%
    # row 2: flat -> excluded from trades entirely, regardless of the (irrelevant) move
    # row 3: long, -0.05% move -> gross = -0.05%
    assert result.n_rows == 4
    assert result.n_trades == 3
    assert result.gross_pnl_pct == pytest.approx(0.10 + 0.10 - 0.05)
    assert result.cost_pct == pytest.approx(3 * cost_per_trade_pct)
    assert result.net_pnl_pct == pytest.approx(0.10 + 0.10 - 0.05 - 3 * cost_per_trade_pct)
    assert result.win_rate == pytest.approx(2 / 3)  # rows 0,1 net positive; row 3 net negative
    assert len(result.per_trade_net_pnl_pct) == 3


def test_simulate_trades_all_flat_reports_zero_trades_without_dividing_by_zero():
    positions = np.zeros(5, dtype=int)
    pd_lead_pct = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
    spread = np.full(5, 0.0002)
    price = np.full(5, 1.10)

    result = simulate_trades(positions, pd_lead_pct, spread, price)
    assert result.n_trades == 0
    assert result.win_rate == 0.0
    assert result.gross_pnl_pct == 0.0
    assert result.net_pnl_pct == 0.0
    assert len(result.per_trade_net_pnl_pct) == 0


def test_simulate_trades_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        simulate_trades(np.array([1, 0]), np.array([0.1]), np.array([0.0002, 0.0002]), np.array([1.1, 1.1]))
