from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from forex_strategy.backtest import (
    position_size_from_realized_volatility,
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

    result = simulate_trades(positions, pd_lead_pct, pd_lead_pct, spread, price)

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


def test_simulate_trades_selects_the_true_side_for_a_wrong_direction_prediction():
    """The core fix: long_raw_return_pct and short_raw_return_pct are each side's
    OWN true outcome, and simulate_trades must select the one matching what was
    actually predicted -- not always use whichever side happens to be passed
    first, and not conflate the two. Row 0 predicts LONG but the long and short
    races genuinely disagree (long really lost -0.4%, short really would have won
    +0.2% raw); row 1 predicts SHORT with the same disagreement. Using the wrong
    side's value here would give the exact opposite of the correct answer."""
    positions = np.array([1, -1])
    long_raw = np.array([-0.4, -0.4])
    short_raw = np.array([0.2, 0.2])
    spread = np.zeros(2)
    price = np.array([1.10, 1.10])

    result = simulate_trades(positions, long_raw, short_raw, spread, price)

    # row 0: long position, must use long_raw (-0.4) -> gross = (+1)*(-0.4) = -0.4
    # row 1: short position, must use short_raw (0.2) -> gross = (-1)*(0.2) = -0.2
    np.testing.assert_allclose(result.per_trade_net_pnl_pct, [-0.4, -0.2])


def test_simulate_trades_all_flat_reports_zero_trades_without_dividing_by_zero():
    positions = np.zeros(5, dtype=int)
    pd_lead_pct = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
    spread = np.full(5, 0.0002)
    price = np.full(5, 1.10)

    result = simulate_trades(positions, pd_lead_pct, pd_lead_pct, spread, price)
    assert result.n_trades == 0
    assert result.win_rate == 0.0
    assert result.gross_pnl_pct == 0.0
    assert result.net_pnl_pct == 0.0
    assert len(result.per_trade_net_pnl_pct) == 0


def test_simulate_trades_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        simulate_trades(
            np.array([1, 0]), np.array([0.1]), np.array([0.1]),
            np.array([0.0002, 0.0002]), np.array([1.1, 1.1]),
        )


def test_position_size_from_realized_volatility_scales_inversely_with_recent_volatility():
    realized_volatility = np.array([0.001, 0.002, 0.004])
    sizes = position_size_from_realized_volatility(realized_volatility, target_volatility=0.002, max_size=2.0)
    np.testing.assert_allclose(sizes, [2.0, 1.0, 0.5])  # row 0 would be 2.0 exactly, clipped at max_size anyway


def test_position_size_from_realized_volatility_clips_at_max_size():
    sizes = position_size_from_realized_volatility(np.array([0.0001]), target_volatility=0.002, max_size=1.5)
    np.testing.assert_allclose(sizes, [1.5])


def test_position_size_from_realized_volatility_zero_volatility_clips_to_max_size_without_dividing_by_zero():
    sizes = position_size_from_realized_volatility(np.array([0.0]), target_volatility=0.002, max_size=1.5)
    np.testing.assert_allclose(sizes, [1.5])


def test_position_size_from_realized_volatility_rejects_negative_values():
    with pytest.raises(ValueError, match="non-negative"):
        position_size_from_realized_volatility(np.array([-0.001]), target_volatility=0.002)


def test_simulate_trades_position_size_scales_pnl_and_cost_proportionally():
    positions = np.array([1, 1])
    pd_lead_pct = np.array([1.0, 1.0])
    spread = np.array([0.0002, 0.0002])
    price = np.array([1.10, 1.10])

    full_size = simulate_trades(positions, pd_lead_pct, pd_lead_pct, spread, price)
    half_size = simulate_trades(
        positions, pd_lead_pct, pd_lead_pct, spread, price, position_size=np.array([0.5, 0.5]),
    )

    assert half_size.gross_pnl_pct == pytest.approx(full_size.gross_pnl_pct * 0.5)
    assert half_size.cost_pct == pytest.approx(full_size.cost_pct * 0.5)
    assert half_size.net_pnl_pct == pytest.approx(full_size.net_pnl_pct * 0.5)


def test_simulate_trades_rejects_mismatched_position_size_length():
    with pytest.raises(ValueError, match="position_size"):
        simulate_trades(
            np.array([1, 0]), np.array([0.1, 0.2]), np.array([0.1, 0.2]),
            np.array([0.0002, 0.0002]), np.array([1.1, 1.1]),
            position_size=np.array([1.0]),
        )


def test_simulate_trades_rejects_negative_position_size():
    """position_size is a magnitude scaler, not a direction flip -- the long/short
    side selection is keyed off `positions`' own sign, not `positions *
    position_size`'s, so a negative entry would silently price a row using the
    wrong side's true outcome. Rejected outright instead of allowed to corrupt
    pricing quietly."""
    with pytest.raises(ValueError, match="non-negative"):
        simulate_trades(
            np.array([1, -1]), np.array([0.1, 0.2]), np.array([0.1, 0.2]),
            np.array([0.0002, 0.0002]), np.array([1.1, 1.1]),
            position_size=np.array([1.0, -0.5]),
        )


def test_simulate_trades_charges_swap_only_when_a_rollover_is_crossed():
    positions = np.array([1, 1])
    pd_lead_pct = np.array([0.0, 0.0])  # isolate swap's effect, no directional P&L
    spread = np.zeros(2)
    price = np.array([1.10, 1.10])
    entry = np.array([_ts(2024, 1, 10, 12, 0), _ts(2024, 1, 10, 12, 0)])
    exit_ = np.array([_ts(2024, 1, 10, 15, 0), _ts(2024, 1, 10, 20, 0)])  # row 0: no rollover, row 1: one rollover

    result = simulate_trades(
        positions, pd_lead_pct, pd_lead_pct, spread, price,
        entry_timestamp=entry, long_exit_timestamp=exit_, short_exit_timestamp=exit_,
        long_swap_cost_pct_per_night=0.05,
    )

    assert result.cost_pct == pytest.approx(0.05)  # only row 1 crosses a rollover
    assert result.net_pnl_pct == pytest.approx(-0.05)


def test_simulate_trades_requires_timestamps_for_swap_cost():
    with pytest.raises(ValueError, match="entry_timestamp"):
        simulate_trades(
            np.array([1]), np.array([0.1]), np.array([0.1]), np.array([0.0]), np.array([1.1]),
            long_swap_cost_pct_per_night=0.05,
        )


def test_simulate_trades_requires_timestamps_for_flatten_before_rollover():
    with pytest.raises(ValueError, match="entry_timestamp"):
        simulate_trades(
            np.array([1]), np.array([0.1]), np.array([0.1]), np.array([0.0]), np.array([1.1]),
            flatten_before_rollover=True,
        )


def test_simulate_trades_flatten_before_rollover_skips_crossing_trades_instead_of_charging_swap():
    positions = np.array([1, 1])
    pd_lead_pct = np.array([1.0, 1.0])
    spread = np.zeros(2)
    price = np.array([1.10, 1.10])
    entry = np.array([_ts(2024, 1, 10, 12, 0), _ts(2024, 1, 10, 12, 0)])
    exit_ = np.array([_ts(2024, 1, 10, 15, 0), _ts(2024, 1, 10, 20, 0)])  # row 0: no rollover, row 1: one rollover

    result = simulate_trades(
        positions, pd_lead_pct, pd_lead_pct, spread, price,
        entry_timestamp=entry, long_exit_timestamp=exit_, short_exit_timestamp=exit_,
        long_swap_cost_pct_per_night=0.05, flatten_before_rollover=True,
    )

    assert result.n_trades == 1  # row 1 was flattened, not held through the rollover
    assert result.n_flattened_for_rollover == 1
    assert result.net_pnl_pct == pytest.approx(1.0)  # row 0 only, no swap ever charged


def test_simulate_trades_charges_the_long_rate_for_a_long_only_position():
    positions = np.array([1, 1])
    pd_lead_pct = np.array([0.0, 0.0])
    spread = np.zeros(2)
    price = np.array([1.10, 1.10])
    entry = np.array([_ts(2024, 1, 10, 12, 0), _ts(2024, 1, 10, 12, 0)])
    exit_ = np.array([_ts(2024, 1, 10, 20, 0), _ts(2024, 1, 10, 20, 0)])  # both cross one rollover

    result = simulate_trades(
        positions, pd_lead_pct, pd_lead_pct, spread, price,
        entry_timestamp=entry, long_exit_timestamp=exit_, short_exit_timestamp=exit_,
        long_swap_cost_pct_per_night=0.05, short_swap_cost_pct_per_night=999.0,  # would blow up cost if misapplied
    )

    assert result.cost_pct == pytest.approx(0.10)  # 2 rows * 0.05, the long rate only


def test_simulate_trades_charges_the_short_rate_for_a_short_only_position():
    positions = np.array([-1, -1])
    pd_lead_pct = np.array([0.0, 0.0])
    spread = np.zeros(2)
    price = np.array([1.10, 1.10])
    entry = np.array([_ts(2024, 1, 10, 12, 0), _ts(2024, 1, 10, 12, 0)])
    exit_ = np.array([_ts(2024, 1, 10, 20, 0), _ts(2024, 1, 10, 20, 0)])

    result = simulate_trades(
        positions, pd_lead_pct, pd_lead_pct, spread, price,
        entry_timestamp=entry, long_exit_timestamp=exit_, short_exit_timestamp=exit_,
        long_swap_cost_pct_per_night=999.0, short_swap_cost_pct_per_night=0.03,
    )

    assert result.cost_pct == pytest.approx(0.06)  # 2 rows * 0.03, the short rate only


def test_simulate_trades_selects_swap_rate_per_row_for_mixed_long_and_short_positions():
    positions = np.array([1, -1, 0])
    pd_lead_pct = np.array([0.0, 0.0, 0.0])
    spread = np.zeros(3)
    price = np.array([1.10, 1.10, 1.10])
    entry = np.array([_ts(2024, 1, 10, 12, 0)] * 3)
    exit_ = np.array([_ts(2024, 1, 10, 20, 0)] * 3)  # all cross one rollover

    result = simulate_trades(
        positions, pd_lead_pct, pd_lead_pct, spread, price,
        entry_timestamp=entry, long_exit_timestamp=exit_, short_exit_timestamp=exit_,
        long_swap_cost_pct_per_night=0.05, short_swap_cost_pct_per_night=0.03,
    )

    # row 0 (long): 0.05, row 1 (short): 0.03, row 2 (flat): excluded entirely
    assert result.n_trades == 2
    assert result.cost_pct == pytest.approx(0.05 + 0.03)


def test_simulate_trades_selects_exit_timestamp_matching_each_rows_own_side():
    """Rollover-crossing counts must use each row's OWN side's exit timestamp, not
    one shared exit_timestamp -- row 0 (long) only crosses a rollover via
    long_exit_timestamp; row 1 (short) only crosses one via short_exit_timestamp.
    Using the wrong side's timestamp here would flip which row gets charged."""
    positions = np.array([1, -1])
    pd_lead_pct = np.array([0.0, 0.0])
    spread = np.zeros(2)
    price = np.array([1.10, 1.10])
    entry = np.array([_ts(2024, 1, 10, 12, 0), _ts(2024, 1, 10, 12, 0)])
    # row 0: long's own exit crosses a rollover, short's own exit doesn't
    # row 1: short's own exit crosses a rollover, long's own exit doesn't
    long_exit = np.array([_ts(2024, 1, 10, 20, 0), _ts(2024, 1, 10, 15, 0)])
    short_exit = np.array([_ts(2024, 1, 10, 15, 0), _ts(2024, 1, 10, 20, 0)])

    result = simulate_trades(
        positions, pd_lead_pct, pd_lead_pct, spread, price,
        entry_timestamp=entry, long_exit_timestamp=long_exit, short_exit_timestamp=short_exit,
        long_swap_cost_pct_per_night=0.05, short_swap_cost_pct_per_night=0.03,
    )

    assert result.cost_pct == pytest.approx(0.05 + 0.03)  # row 0 charged long's rate, row 1 short's rate
