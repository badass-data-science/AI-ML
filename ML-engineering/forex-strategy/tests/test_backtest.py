from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from forex_strategy.backtest import (
    position_size_from_predicted_volatility_class,
    predicted_classes_to_positions,
    simulate_trades,
)

_NY = ZoneInfo("America/New_York")


def _ts(*args) -> float:
    return datetime.datetime(*args, tzinfo=_NY).timestamp()


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


def test_position_size_from_predicted_volatility_class_maps_terciles_to_sizes():
    sizes = position_size_from_predicted_volatility_class(np.array([0, 1, 2, 0]))
    np.testing.assert_array_equal(sizes, [1.0, 0.6, 0.3, 1.0])


def test_position_size_from_predicted_volatility_class_accepts_custom_sizes():
    sizes = position_size_from_predicted_volatility_class(np.array([0, 2]), size_by_class=(2.0, 1.0, 0.0))
    np.testing.assert_array_equal(sizes, [2.0, 0.0])


def test_position_size_from_predicted_volatility_class_rejects_out_of_range():
    with pytest.raises(ValueError, match="0, 1, 2"):
        position_size_from_predicted_volatility_class(np.array([0, 3]))


def test_simulate_trades_position_size_scales_pnl_and_cost_proportionally():
    positions = np.array([1, 1])
    pd_lead_pct = np.array([1.0, 1.0])
    spread = np.array([0.0002, 0.0002])
    price = np.array([1.10, 1.10])

    full_size = simulate_trades(positions, pd_lead_pct, spread, price)
    half_size = simulate_trades(positions, pd_lead_pct, spread, price, position_size=np.array([0.5, 0.5]))

    assert half_size.gross_pnl_pct == pytest.approx(full_size.gross_pnl_pct * 0.5)
    assert half_size.cost_pct == pytest.approx(full_size.cost_pct * 0.5)
    assert half_size.net_pnl_pct == pytest.approx(full_size.net_pnl_pct * 0.5)


def test_simulate_trades_rejects_mismatched_position_size_length():
    with pytest.raises(ValueError, match="position_size"):
        simulate_trades(
            np.array([1, 0]), np.array([0.1, 0.2]), np.array([0.0002, 0.0002]), np.array([1.1, 1.1]),
            position_size=np.array([1.0]),
        )


def test_simulate_trades_charges_swap_only_when_a_rollover_is_crossed():
    positions = np.array([1, 1])
    pd_lead_pct = np.array([0.0, 0.0])  # isolate swap's effect, no directional P&L
    spread = np.zeros(2)
    price = np.array([1.10, 1.10])
    entry = np.array([_ts(2024, 1, 10, 12, 0), _ts(2024, 1, 10, 12, 0)])
    exit_ = np.array([_ts(2024, 1, 10, 15, 0), _ts(2024, 1, 10, 20, 0)])  # row 0: no rollover, row 1: one rollover

    result = simulate_trades(
        positions, pd_lead_pct, spread, price,
        entry_timestamp=entry, exit_timestamp=exit_, swap_cost_pct_per_night=0.05,
    )

    assert result.cost_pct == pytest.approx(0.05)  # only row 1 crosses a rollover
    assert result.net_pnl_pct == pytest.approx(-0.05)


def test_simulate_trades_requires_timestamps_for_swap_cost():
    with pytest.raises(ValueError, match="entry_timestamp"):
        simulate_trades(
            np.array([1]), np.array([0.1]), np.array([0.0]), np.array([1.1]), swap_cost_pct_per_night=0.05,
        )


def test_simulate_trades_requires_timestamps_for_flatten_before_rollover():
    with pytest.raises(ValueError, match="entry_timestamp"):
        simulate_trades(
            np.array([1]), np.array([0.1]), np.array([0.0]), np.array([1.1]), flatten_before_rollover=True,
        )


def test_simulate_trades_flatten_before_rollover_skips_crossing_trades_instead_of_charging_swap():
    positions = np.array([1, 1])
    pd_lead_pct = np.array([1.0, 1.0])
    spread = np.zeros(2)
    price = np.array([1.10, 1.10])
    entry = np.array([_ts(2024, 1, 10, 12, 0), _ts(2024, 1, 10, 12, 0)])
    exit_ = np.array([_ts(2024, 1, 10, 15, 0), _ts(2024, 1, 10, 20, 0)])  # row 0: no rollover, row 1: one rollover

    result = simulate_trades(
        positions, pd_lead_pct, spread, price,
        entry_timestamp=entry, exit_timestamp=exit_,
        swap_cost_pct_per_night=0.05, flatten_before_rollover=True,
    )

    assert result.n_trades == 1  # row 1 was flattened, not held through the rollover
    assert result.n_flattened_for_rollover == 1
    assert result.net_pnl_pct == pytest.approx(1.0)  # row 0 only, no swap ever charged
