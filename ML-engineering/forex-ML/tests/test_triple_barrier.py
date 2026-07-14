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


def test_triple_barrier_labels_hits_long_profit_take():
    price = np.array([100.0, 100.5, 101.5, 99.0])
    spread = np.zeros(4)
    timestamp = _ts(2024, 1, 10, 6, 0) + np.arange(4) * 3600.0  # 6am NY, well before rollover

    result = triple_barrier_labels(
        price, spread, timestamp, profit_take_pct=1.0, stop_loss_pct=1.0, max_holding_bars=3,
    )

    assert result.label[0] == 1
    assert result.exit_bar_offset[0] == 2
    assert result.net_return_pct[0] == pytest.approx(1.5)


def test_triple_barrier_labels_hits_short_profit_take():
    """Short's own race independently confirms profitability here (zero cost, a
    genuine 1.5% drop clearing the 1.0% threshold) -- NOT just "long's stop-loss
    fired," which is what this label used to mean before the bidirectional
    redesign. See test_short_signal_requires_short_side_to_actually_clear_its_own_cost
    below for the regression case that distinguishes the two."""
    price = np.array([100.0, 99.7, 98.5])
    spread = np.zeros(3)
    timestamp = _ts(2024, 1, 10, 6, 0) + np.arange(3) * 3600.0

    result = triple_barrier_labels(
        price, spread, timestamp, profit_take_pct=1.0, stop_loss_pct=1.0, max_holding_bars=2,
    )

    assert result.label[0] == -1
    assert result.exit_bar_offset[0] == 2
    # net_return_pct is now the WINNING side's own net return -- short's, which is
    # positive here (a real gain), not long's (which would show its own -1.5% loss).
    assert result.net_return_pct[0] == pytest.approx(1.5)
    # raw_return_pct is always the objective price return, sign-consistent
    # regardless of which side won -- price genuinely fell 1.5%.
    assert result.raw_return_pct[0] == pytest.approx(-1.5)


def test_long_and_short_raw_return_pct_are_both_persisted_even_when_they_disagree():
    """The regression case for the backtest-mispricing fix: long_raw_return_pct/
    long_exit_bar_offset and short_raw_return_pct/short_exit_bar_offset must be the
    TRUE outcome of each side's own race, independently -- not just whichever side
    won. Constructed so the two races resolve at genuinely DIFFERENT bars with
    genuinely DIFFERENT raw returns (short's own credit-earning swap lets it clear
    its profit-take early at bar 2 off a -0.2% move; long's race keeps running
    until its own stop-loss fires later, at bar 3, off a bigger -0.4% move) --
    before this fix, a model predicting the WRONG direction (long, when short
    actually won) would have been priced using short's -0.2% raw return as a
    stand-in, instead of long's true -0.4%."""
    price = np.array([100.0, 99.8, 99.8, 99.6, 99.9, 99.5])
    spread = np.zeros(6)
    timestamp = np.array([_ts(2024, 1, 10 + d, 12, 0) for d in range(6)])

    result = triple_barrier_labels(
        price, spread, timestamp, profit_take_pct=0.3, stop_loss_pct=0.3, max_holding_bars=5,
        long_swap_cost_pct_per_night=0.0, short_swap_cost_pct_per_night=-0.05,
    )

    assert result.label[0] == -1  # short wins
    # the merged/single-winner view (pre-existing fields) reflects short's outcome
    assert result.exit_bar_offset[0] == 2
    assert result.raw_return_pct[0] == pytest.approx(-0.2)
    # but long's own true race outcome -- what a wrong-direction long bet would
    # really have realized -- is still independently available, and differs
    assert result.long_exit_bar_offset[0] == 3
    assert result.long_raw_return_pct[0] == pytest.approx(-0.4)
    assert result.short_exit_bar_offset[0] == 2
    assert result.short_raw_return_pct[0] == pytest.approx(-0.2)


def test_short_signal_requires_short_side_to_actually_clear_its_own_cost():
    """The key regression case: a drop big enough to stop out a long does NOT,
    by itself, mean a short would have been profitable -- the short must ALSO
    clear its own cost. Concretely (matching the design's worked example):
    profit_take_pct=stop_loss_pct=0.3, entry_cost_pct=0.02 (from spread), one
    rollover crossed, long_swap=-0.01/night (a small credit), short_swap=0.005/
    night (a small cost), and a 0.30% drop.

    long_net = -0.30 - 0.02 - (-0.01) = -0.31 <= -0.30 -> long's stop-loss fires.
    short_net = -(-0.30) - 0.02 - 0.005 = 0.275 < 0.30 -> short's OWN profit-take
    does NOT clear.

    Under the old (pre-redesign) proxy, "long's stop-loss fired" alone would have
    been mislabeled as a short signal (-1). The fix: label 0 (flat), not -1.
    """
    price = np.array([100.0, 99.70])
    spread = np.array([0.02, 0.02])
    timestamp = np.array([_ts(2024, 1, 10, 12, 0), _ts(2024, 1, 10, 20, 0)])  # crosses one 5pm rollover

    result = triple_barrier_labels(
        price, spread, timestamp, profit_take_pct=0.3, stop_loss_pct=0.3, max_holding_bars=1,
        long_swap_cost_pct_per_night=-0.01, short_swap_cost_pct_per_night=0.005,
    )

    assert result.label[0] == 0


def test_short_signal_fires_once_the_drop_is_big_enough_to_clear_short_cost_too():
    """Same cost structure as the regression test above, but with a bigger drop
    (0.50%) that clears BOTH long's stop-loss AND short's own cost-adjusted
    profit-take -- this time a genuine short signal."""
    price = np.array([100.0, 99.50])
    spread = np.array([0.02, 0.02])
    timestamp = np.array([_ts(2024, 1, 10, 12, 0), _ts(2024, 1, 10, 20, 0)])

    result = triple_barrier_labels(
        price, spread, timestamp, profit_take_pct=0.3, stop_loss_pct=0.3, max_holding_bars=1,
        long_swap_cost_pct_per_night=-0.01, short_swap_cost_pct_per_night=0.005,
    )

    assert result.label[0] == -1
    assert result.net_return_pct[0] == pytest.approx(0.5 - 0.02 - 0.005)


def test_asymmetric_swap_rates_affect_only_their_own_side():
    """A 0.35% drop, zero spread, one rollover crossed. With both swap costs at
    0.0, short's own profit-take clears (0.35 >= 0.3) -- a genuine short signal.
    Raising ONLY short_swap_cost_pct_per_night (leaving long's swap untouched)
    should be able to flip that specific outcome to flat, without changing
    anything about how the long side is evaluated."""
    price = np.array([100.0, 99.65])
    spread = np.zeros(2)
    timestamp = np.array([_ts(2024, 1, 10, 12, 0), _ts(2024, 1, 10, 20, 0)])

    baseline = triple_barrier_labels(
        price, spread, timestamp, profit_take_pct=0.3, stop_loss_pct=0.3, max_holding_bars=1,
        long_swap_cost_pct_per_night=0.0, short_swap_cost_pct_per_night=0.0,
    )
    assert baseline.label[0] == -1  # short's own 0.35% >= 0.3% profit-take, no cost drag

    higher_short_cost = triple_barrier_labels(
        price, spread, timestamp, profit_take_pct=0.3, stop_loss_pct=0.3, max_holding_bars=1,
        long_swap_cost_pct_per_night=0.0, short_swap_cost_pct_per_night=0.1,
    )
    # 0.35 - 0.1 = 0.25 < 0.3 -- short's own profit-take no longer clears, and
    # long's stop-loss doesn't count as a win either, so this becomes flat.
    assert higher_short_cost.label[0] == 0


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
        max_holding_bars=2, long_swap_cost_pct_per_night=0.0,
    )
    with_swap = triple_barrier_labels(
        price, spread, timestamp, profit_take_pct=10.0, stop_loss_pct=10.0,
        max_holding_bars=2, long_swap_cost_pct_per_night=0.2,
    )

    assert no_swap.label[0] == 0 and with_swap.label[0] == 0  # both time out
    assert no_swap.net_return_pct[0] == pytest.approx(0.3)
    assert with_swap.net_return_pct[0] == pytest.approx(0.3 - 0.2)  # exactly one rollover charged


def test_different_bar_ties_are_resolved_by_whichever_resolves_first():
    """Both long's and short's profit-takes are individually reachable within the
    window, but at different bars: long's clears the (small) 0.5% threshold on a
    moderate rise at bar 1; short's own profit-take only clears later, at bar 2,
    once price reverses hard enough. Long resolved first, so long wins overall --
    this is the "earlier bar wins" tie-break, exercised on a genuinely different-
    bar case (not the same-bar case, which has its own test below)."""
    price = np.array([100.0, 100.6, 99.0])
    spread = np.zeros(3)
    timestamp = _ts(2024, 1, 10, 6, 0) + np.arange(3) * 3600.0

    result = triple_barrier_labels(
        price, spread, timestamp, profit_take_pct=0.5, stop_loss_pct=5.0, max_holding_bars=2,
    )

    assert result.label[0] == 1
    assert result.exit_bar_offset[0] == 1
    assert result.net_return_pct[0] == pytest.approx(0.6)


def test_same_bar_double_fire_is_resolved_by_the_explicit_tie_break_not_assumed_impossible():
    """A same-bar double-fire is only impossible under "normal" cost/swap
    magnitudes -- the algebra (long_net + short_net = -2*entry_cost_pct -
    (long_swap + short_swap)*rollovers) shows it's reachable if both swap rates
    are configured as large enough credits. Price doesn't even move here (raw
    return is exactly 0%); a large offsetting swap credit on both sides is enough
    to clear a tiny profit-take threshold for BOTH sides at the same (only) bar.
    The explicit tie-break rule (long wins) must fire here, not an unenforced
    "this can't happen" assumption."""
    price = np.array([100.0, 100.0])
    spread = np.zeros(2)
    timestamp = np.array([_ts(2024, 1, 10, 12, 0), _ts(2024, 1, 10, 20, 0)])  # one rollover crossed

    result = triple_barrier_labels(
        price, spread, timestamp, profit_take_pct=0.001, stop_loss_pct=5.0, max_holding_bars=1,
        long_swap_cost_pct_per_night=-1.0, short_swap_cost_pct_per_night=-1.0,
    )

    assert result.label[0] == 1  # long wins the tie
    assert result.exit_bar_offset[0] == 1
    assert result.net_return_pct[0] == pytest.approx(1.0)
    assert result.raw_return_pct[0] == pytest.approx(0.0)


def test_flat_row_reference_is_whichever_race_resolves_first_not_always_long():
    """Short's own stop-loss fires early (bar 1, price rises against it), while
    long's race continues and only times out later (bar 3). Neither side wins
    (short stopped, long timed out), so the row is flat -- but the reported
    exit_bar_offset/net_return_pct should reference SHORT's race (resolved at
    bar 1), not default to long's (which resolves later, at bar 3)."""
    price = np.array([100.0, 100.5, 100.3, 100.4])
    spread = np.zeros(4)
    timestamp = _ts(2024, 1, 10, 6, 0) + np.arange(4) * 3600.0

    result = triple_barrier_labels(
        price, spread, timestamp, profit_take_pct=5.0, stop_loss_pct=0.3, max_holding_bars=3,
    )

    assert result.label[0] == 0
    assert result.exit_bar_offset[0] == 1
    assert result.net_return_pct[0] == pytest.approx(-0.5)
    assert result.raw_return_pct[0] == pytest.approx(0.5)


def test_triple_barrier_labels_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        triple_barrier_labels(np.array([1.0, 2.0]), np.array([0.0]), np.array([0.0, 1.0]), 1.0, 1.0, 1)


def test_triple_barrier_labels_rejects_non_positive_thresholds():
    with pytest.raises(ValueError, match="positive"):
        triple_barrier_labels(np.array([1.0, 2.0]), np.zeros(2), np.zeros(2), 0.0, 1.0, 1)


def test_triple_barrier_labels_rejects_too_few_rows():
    with pytest.raises(ValueError, match="Need more than"):
        triple_barrier_labels(np.array([1.0, 2.0]), np.zeros(2), np.zeros(2), 1.0, 1.0, max_holding_bars=5)


def test_triple_barrier_labels_rejects_negative_spread():
    """Needed for the same-bar tie-break argument to actually hold -- a negative
    spread (entry_cost_pct < 0) is one of the two ways a same-bar double-fire
    becomes reachable, so it's rejected outright rather than silently accepted."""
    with pytest.raises(ValueError, match="non-negative"):
        triple_barrier_labels(np.array([1.0, 2.0]), np.array([-0.1, -0.1]), np.zeros(2), 1.0, 1.0, 1)


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
    # long wins here, so the merged view and long's own view agree
    assert out.iloc[0]["long_exit_bar_offset"] == 2
    assert out.iloc[0]["long_raw_return_pct"] == pytest.approx(1.5)
    # short's own race outcome is independently available too, win or lose
    assert "short_exit_bar_offset" in out.columns
    assert "short_raw_return_pct" in out.columns


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
        df, profit_take_pct=0.1, stop_loss_pct=0.1, max_holding_bars=4,
        long_swap_cost_pct_per_night=0.01, short_swap_cost_pct_per_night=0.01,
    )

    assert len(out) == len(df) - 4
    assert set(out["label"].unique()) <= {-1, 0, 1}
