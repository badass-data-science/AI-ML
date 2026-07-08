from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from forex_ml.config import FeatureParams
from forex_ml.data.triple_barrier import (
    count_rollovers_crossed,
    triple_barrier_labels,
    triple_barrier_labels_from_frame,
)
from forex_ml.flows.prepare_data_flow import engineer_and_save_task
from forex_ml.paths import non_time_series_parquet_path, pair_key

_NY = ZoneInfo("America/New_York")


def _ts(*args) -> float:
    return datetime.datetime(*args, tzinfo=_NY).timestamp()


class TestCountRolloversCrossed:
    def test_same_day_before_5pm_is_zero(self):
        assert count_rollovers_crossed(_ts(2024, 1, 10, 12, 0), _ts(2024, 1, 10, 16, 0)) == 0

    def test_spanning_5pm_is_one(self):
        assert count_rollovers_crossed(_ts(2024, 1, 10, 12, 0), _ts(2024, 1, 10, 18, 0)) == 1

    def test_exactly_at_5pm_counts_as_crossed(self):
        assert count_rollovers_crossed(_ts(2024, 1, 10, 12, 0), _ts(2024, 1, 10, 17, 0)) == 1

    def test_spanning_two_nights_is_two(self):
        assert count_rollovers_crossed(_ts(2024, 1, 10, 12, 0), _ts(2024, 1, 12, 12, 0)) == 2

    def test_dst_spring_forward_still_counts_one_rollover_per_calendar_night(self):
        # 2024-03-10 is when US DST begins -- still exactly one 5pm boundary per
        # night regardless of the clock jump, since the rollover is defined in
        # local NY wall-clock time, not a fixed UTC offset.
        assert count_rollovers_crossed(_ts(2024, 3, 9, 12, 0), _ts(2024, 3, 10, 20, 0)) == 2


def test_triple_barrier_labels_hits_profit_take():
    price = np.array([100.0, 100.5, 101.5, 99.0])
    spread = np.zeros(4)
    timestamp = _ts(2024, 1, 10, 6, 0) + np.arange(4) * 3600.0  # 6am NY, well before rollover

    result = triple_barrier_labels(
        price, spread, timestamp, profit_take_pct=1.0, stop_loss_pct=1.0, max_holding_bars=3,
    )

    assert result.label[0] == 1
    assert result.exit_bar_offset[0] == 2
    assert result.net_return_pct[0] == pytest.approx(1.5)


def test_triple_barrier_labels_hits_stop_loss():
    price = np.array([100.0, 99.7, 98.5])
    spread = np.zeros(3)
    timestamp = _ts(2024, 1, 10, 6, 0) + np.arange(3) * 3600.0

    result = triple_barrier_labels(
        price, spread, timestamp, profit_take_pct=1.0, stop_loss_pct=1.0, max_holding_bars=2,
    )

    assert result.label[0] == -1
    assert result.exit_bar_offset[0] == 2
    assert result.net_return_pct[0] == pytest.approx(-1.5)


def test_triple_barrier_labels_times_out_when_neither_barrier_is_hit():
    price = np.array([100.0, 100.2, 100.3])
    spread = np.zeros(3)
    timestamp = _ts(2024, 1, 10, 6, 0) + np.arange(3) * 3600.0

    result = triple_barrier_labels(
        price, spread, timestamp, profit_take_pct=1.0, stop_loss_pct=1.0, max_holding_bars=2,
    )

    assert result.label[0] == 0
    assert result.exit_bar_offset[0] == 2
    assert result.net_return_pct[0] == pytest.approx(0.3)


def test_triple_barrier_labels_is_cost_aware_a_raw_hit_can_become_a_timeout():
    """The whole point of this module: a +1.5% raw move clears a 1.0% profit-take
    barrier with no cost, but the same identical price path no longer clears it
    once a 1% round-trip spread is charged."""
    price = np.array([100.0, 101.5])
    timestamp = _ts(2024, 1, 10, 6, 0) + np.arange(2) * 3600.0

    zero_cost = triple_barrier_labels(
        price, np.zeros(2), timestamp, profit_take_pct=1.0, stop_loss_pct=5.0, max_holding_bars=1,
    )
    assert zero_cost.label[0] == 1

    with_cost = triple_barrier_labels(
        price, np.array([1.0, 1.0]), timestamp, profit_take_pct=1.0, stop_loss_pct=5.0, max_holding_bars=1,
    )
    assert with_cost.label[0] == 0
    assert with_cost.net_return_pct[0] == pytest.approx(0.5)
    # raw_return_pct is the SAME +1.5% either way -- it's the pre-cost move, so a
    # downstream backtest that charges its own cost doesn't double-count spread/swap.
    assert zero_cost.raw_return_pct[0] == pytest.approx(1.5)
    assert with_cost.raw_return_pct[0] == pytest.approx(1.5)


def test_triple_barrier_labels_charges_swap_only_once_per_rollover_actually_crossed():
    # profit_take/stop_loss set impossibly wide (10%) so every row times out at
    # max_holding_bars regardless of swap -- isolates swap's effect on net_return_pct.
    price = np.array([100.0, 100.3, 100.3])
    spread = np.zeros(3)
    # noon NY -> +3h (still before 5pm) -> +8h from entry (crosses exactly one rollover)
    timestamp = np.array([_ts(2024, 1, 10, 12, 0), _ts(2024, 1, 10, 15, 0), _ts(2024, 1, 10, 20, 0)])

    no_swap = triple_barrier_labels(
        price, spread, timestamp, profit_take_pct=10.0, stop_loss_pct=10.0,
        max_holding_bars=2, swap_cost_pct_per_night=0.0,
    )
    with_swap = triple_barrier_labels(
        price, spread, timestamp, profit_take_pct=10.0, stop_loss_pct=10.0,
        max_holding_bars=2, swap_cost_pct_per_night=0.2,
    )

    assert no_swap.label[0] == 0 and with_swap.label[0] == 0  # both time out
    assert no_swap.net_return_pct[0] == pytest.approx(0.3)
    assert with_swap.net_return_pct[0] == pytest.approx(0.3 - 0.2)  # exactly one rollover charged


def test_triple_barrier_labels_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        triple_barrier_labels(np.array([1.0, 2.0]), np.array([0.0]), np.array([0.0, 1.0]), 1.0, 1.0, 1)


def test_triple_barrier_labels_rejects_non_positive_thresholds():
    with pytest.raises(ValueError, match="positive"):
        triple_barrier_labels(np.array([1.0, 2.0]), np.zeros(2), np.zeros(2), 0.0, 1.0, 1)


def test_triple_barrier_labels_rejects_too_few_rows():
    with pytest.raises(ValueError, match="Need more than"):
        triple_barrier_labels(np.array([1.0, 2.0]), np.zeros(2), np.zeros(2), 1.0, 1.0, max_holding_bars=5)


def test_triple_barrier_labels_from_frame_appends_expected_columns_and_keeps_others():
    df = pd.DataFrame({
        "instrument": ["EUR/USD"] * 4,
        "mid_close": [100.0, 100.5, 101.5, 99.0],
        "spread_close": [0.0, 0.0, 0.0, 0.0],
        "unix_epoch_s": _ts(2024, 1, 10, 6, 0) + np.arange(4) * 3600.0,
    })

    out = triple_barrier_labels_from_frame(df, profit_take_pct=1.0, stop_loss_pct=1.0, max_holding_bars=3)

    assert len(out) == 1
    assert out.iloc[0]["label"] == 1
    assert out.iloc[0]["exit_bar_offset"] == 2
    assert out.iloc[0]["raw_return_pct"] == pytest.approx(1.5)
    assert out.iloc[0]["instrument"] == "EUR/USD"  # untouched passthrough column


def test_triple_barrier_labels_from_frame_runs_against_real_stage1_output(spark, synthetic_candles, tmp_path):
    params = FeatureParams(
        instruments=["EUR/USD"],
        granularities=["H1"],
        n_back=10,
        lookahead=2,
        ma_lookback_list=[3, 5],
        columns_base=["mid_open", "mid_high", "mid_low", "mid_close", "spread_close", "volume"],
        ma_columns_list=["volatility", "return", "diff_spread_close", "diff_volume"],
        training_and_testing=True,
        min_training_timestamp="2020-01-01T00:00:00",
        output_dir=str(tmp_path),
    )
    engineer_and_save_task(spark, synthetic_candles, "EUR/USD", "H1", params)

    key = pair_key("EUR/USD", "H1", params.n_back, params.lookahead)
    df = pd.read_parquet(non_time_series_parquet_path(str(tmp_path), key)).sort_values("unix_epoch_s")

    out = triple_barrier_labels_from_frame(
        df, profit_take_pct=0.1, stop_loss_pct=0.1, max_holding_bars=4, swap_cost_pct_per_night=0.01,
    )

    assert len(out) == len(df) - 4
    assert set(out["label"].unique()) <= {-1, 0, 1}
