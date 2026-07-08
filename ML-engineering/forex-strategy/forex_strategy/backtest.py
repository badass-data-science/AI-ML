"""Cost-aware backtest: turn a model's per-row class predictions into a trade
decision, and simulate P&L net of spread and (optionally) swap/rollover cost.

Class semantics match forex_ml.data.splitting.TimeSeriesSplitter._compute_outcome:
class 0 = lowest tercile of column_y, class 1 = middle, class 2 = highest tercile.
For a DIRECTIONAL target like pd_lead (% price change), that maps naturally onto
short / no-trade / long. This backtest assumes a directional target -- running it
against a magnitude-only target (volatility_lead, forex-ML's current default) isn't
meaningful on its own, since "high volatility" implies no direction. Volatility is
meant to feed position SIZING on top of a directional decision, not substitute for
one -- see `position_size_from_predicted_volatility_class` below. Callers should
check the source run's logged `column_y` param (see forex-ML's README) before
treating `y_raw` as a directional P&L input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from forex_ml.data.triple_barrier import count_rollovers_crossed


@dataclass
class BacktestResult:
    n_rows: int
    n_trades: int
    n_flattened_for_rollover: int
    win_rate: float
    gross_pnl_pct: float
    cost_pct: float
    net_pnl_pct: float
    per_trade_net_pnl_pct: np.ndarray


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


def position_size_from_predicted_volatility_class(
    predicted_volatility_class: np.ndarray, size_by_class: tuple[float, float, float] = (1.0, 0.6, 0.3),
) -> np.ndarray:
    """Scales position size by a volatility model's predicted tercile: full size for
    a predicted LOW-volatility bar (class 0), progressively smaller for medium
    (class 1) and high (class 2) -- the standard volatility-targeting idea (size
    inversely with risk), applied to forex-ML's ordinal 3-class `volatility_lead`
    prediction rather than a fabricated continuous magnitude the model was never
    trained to output precisely."""
    predicted_volatility_class = np.asarray(predicted_volatility_class)
    if predicted_volatility_class.size and (
        predicted_volatility_class.min() < 0 or predicted_volatility_class.max() > 2
    ):
        raise ValueError("predicted_volatility_class must be in {0, 1, 2}")
    return np.asarray(size_by_class)[predicted_volatility_class]


def simulate_trades(
    positions: np.ndarray,
    pd_lead_pct: np.ndarray,
    spread: np.ndarray,
    price: np.ndarray,
    *,
    position_size: np.ndarray | None = None,
    entry_timestamp: np.ndarray | None = None,
    exit_timestamp: np.ndarray | None = None,
    swap_cost_pct_per_night: float = 0.0,
    flatten_before_rollover: bool = False,
) -> BacktestResult:
    """Spread is charged as the full round-trip cost -- buying at ask and later
    selling at bid (or the reverse, for a short) costs one full spread-width
    relative to the mid price `pd_lead_pct` is measured from. The entry bar's
    spread stands in for both legs since the exit bar's spread isn't available here
    (forex-ML computes `spread_close_lead` in Stage 1 but doesn't currently plumb
    it through Splits).

    Swap/rollover, if `swap_cost_pct_per_night` is nonzero, is charged once per 5pm
    New York rollover boundary actually crossed between `entry_timestamp` and
    `exit_timestamp` (via forex-ML's `count_rollovers_crossed` -- DST-aware, not a
    fixed-UTC approximation), not once per bar held -- an intraday hold usually
    crosses zero rollovers. `flatten_before_rollover=True` implements the "flatten
    by 5pm" rule instead of paying swap: any row whose holding period would cross a
    rollover is forced flat (skipped) rather than held through it and charged --
    both `swap_cost_pct_per_night` and `flatten_before_rollover` require
    `entry_timestamp`/`exit_timestamp`.

    `position_size` (default: all ones) scales both P&L and cost proportionally,
    so a 0.3-size position produces 30% of a full-size position's P&L AND 30% of
    its cost, not a discounted cost at full P&L -- see
    `position_size_from_predicted_volatility_class` for a volatility-gated source.

    `pd_lead_pct` is forex_ml's pd_lead target: 100 * (exit_price - entry_price) /
    entry_price, i.e. already a percentage -- see this module's docstring for why
    that's the only target this makes sense against. `spread`/`price` are the raw
    (non-percentage) price-unit values from Splits.test.
    """
    n = len(positions)
    if not (len(pd_lead_pct) == len(spread) == len(price) == n):
        raise ValueError("positions/pd_lead_pct/spread/price must all be the same length")

    if position_size is None:
        position_size = np.ones(n)
    elif len(position_size) != n:
        raise ValueError("position_size must be the same length as positions")

    needs_timestamps = swap_cost_pct_per_night != 0.0 or flatten_before_rollover
    if needs_timestamps and (entry_timestamp is None or exit_timestamp is None):
        raise ValueError(
            "entry_timestamp and exit_timestamp are required when swap_cost_pct_per_night "
            "is set or flatten_before_rollover is True"
        )
    if needs_timestamps and not (len(entry_timestamp) == len(exit_timestamp) == n):  # type: ignore[arg-type]
        raise ValueError("entry_timestamp/exit_timestamp must be the same length as positions")

    if needs_timestamps:
        n_rollovers = np.array([
            count_rollovers_crossed(e, x) for e, x in zip(entry_timestamp, exit_timestamp)  # type: ignore[arg-type]
        ])
    else:
        n_rollovers = np.zeros(n, dtype=int)

    sized_positions = positions * position_size

    n_flattened_for_rollover = 0
    if flatten_before_rollover:
        would_cross = n_rollovers > 0
        n_flattened_for_rollover = int(((positions != 0) & would_cross).sum())
        sized_positions = np.where(would_cross, 0.0, sized_positions)

    is_trade = sized_positions != 0
    abs_size = np.abs(sized_positions)
    gross_pnl_pct = sized_positions * pd_lead_pct
    spread_cost_pct = np.where(is_trade, abs_size * 100.0 * spread / price, 0.0)
    swap_cost_pct = np.where(is_trade, abs_size * swap_cost_pct_per_night * n_rollovers, 0.0)
    cost_pct = spread_cost_pct + swap_cost_pct
    net_pnl_pct = np.where(is_trade, gross_pnl_pct - cost_pct, 0.0)

    n_trades = int(is_trade.sum())
    per_trade = net_pnl_pct[is_trade]

    return BacktestResult(
        n_rows=n,
        n_trades=n_trades,
        n_flattened_for_rollover=n_flattened_for_rollover,
        win_rate=float((per_trade > 0).mean()) if n_trades else 0.0,
        gross_pnl_pct=float(gross_pnl_pct[is_trade].sum()) if n_trades else 0.0,
        cost_pct=float(cost_pct[is_trade].sum()) if n_trades else 0.0,
        net_pnl_pct=float(per_trade.sum()) if n_trades else 0.0,
        per_trade_net_pnl_pct=per_trade,
    )
