"""Cost-aware backtest: turn a model's per-row class predictions into a trade
decision, and simulate P&L net of spread and (optionally) swap/rollover cost.

Class semantics match forex_ml.data.splitting.TimeSeriesSplitter._label_to_one_hot's
triple-barrier label mapping: class 0 = short's own profit-take independently
fired (label -1), class 1 = neither side's profit-take fired / flat (label 0),
class 2 = long's profit-take fired (label +1) -- so highest class = long signal,
lowest class = short signal, exactly the short/flat/long convention this module's
`predicted_classes_to_positions` already assumes. (Label -1 used to mean "the
long's stop-loss hit," a proxy that didn't verify independent short profitability
-- see triple_barrier.py's module docstring for the bidirectional redesign that
fixed this; the class/position convention here was already correct and needed no
change.) Position SIZING is a
separate concern from the directional call: `position_size_from_realized_volatility`
below scales size against forex-ML's `realized_volatility` passthrough column (a
real, already-observed backward-looking average -- see forex-ML's README), not a
second model's prediction.
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


def position_size_from_realized_volatility(
    realized_volatility: np.ndarray, target_volatility: float, max_size: float = 1.0,
) -> np.ndarray:
    """Standard inverse-volatility-targeting: size = clip(target_volatility /
    realized_volatility, 0, max_size). Scales a trade down when recent realized
    volatility (forex-ML's `realized_volatility` passthrough column -- a fixed
    12-bar trailing average of per-bar high-low range, real already-observed data,
    not a prediction) is running above `target_volatility`, and up (capped at
    `max_size`) when it's running below. A realized_volatility of exactly 0 (e.g. a
    dead/illiquid stretch) sizes at `max_size` rather than dividing by zero -- there
    was no recent risk observed, not infinite headroom, but `max_size` is the
    least-wrong finite answer here."""
    realized_volatility = np.asarray(realized_volatility, dtype=float)
    if realized_volatility.size and realized_volatility.min() < 0:
        raise ValueError("realized_volatility must be non-negative")
    with np.errstate(divide="ignore"):
        size = np.where(realized_volatility > 0, target_volatility / realized_volatility, max_size)
    return np.clip(size, 0.0, max_size)


def simulate_trades(
    positions: np.ndarray,
    long_raw_return_pct: np.ndarray,
    short_raw_return_pct: np.ndarray,
    spread: np.ndarray,
    price: np.ndarray,
    *,
    position_size: np.ndarray | None = None,
    entry_timestamp: np.ndarray | None = None,
    long_exit_timestamp: np.ndarray | None = None,
    short_exit_timestamp: np.ndarray | None = None,
    long_swap_cost_pct_per_night: float = 0.0,
    short_swap_cost_pct_per_night: float = 0.0,
    flatten_before_rollover: bool = False,
) -> BacktestResult:
    """Spread is charged as the full round-trip cost -- buying at ask and later
    selling at bid (or the reverse, for a short) costs one full spread-width
    relative to the mid price returns are measured from. The entry bar's spread
    stands in for both legs since the exit bar's spread isn't available here
    (forex-ML computes `spread_close_lead` in Stage 1 but doesn't currently plumb
    it through Splits).

    `long_raw_return_pct`/`short_raw_return_pct` (and, if swap cost or
    flatten_before_rollover is used, `long_exit_timestamp`/`short_exit_timestamp`)
    are EACH SIDE'S OWN outcome -- see forex_ml.data.triple_barrier.
    TripleBarrierLabels' long_*/short_* fields -- not a single merged "whichever
    side the label was" value. This function selects the side matching each row's
    OWN `positions` sign (long_* for positions > 0, short_* otherwise) before
    computing P&L/cost, so a wrong-direction prediction (positions disagrees with
    what actually won) is priced using the TRUE outcome of the side actually taken,
    not the winning side's outcome as a stand-in. Passing the same array for both
    (e.g. a single merged raw_return_pct) silently reproduces the old, mispricing
    behavior -- pass each side's real value.

    Swap/rollover, if either swap-cost param is nonzero, is charged once per 5pm
    New York rollover boundary actually crossed between `entry_timestamp` and the
    row's own side's exit timestamp (via forex-ML's `count_rollovers_crossed` --
    DST-aware, not a fixed-UTC approximation), not once per bar held -- an
    intraday hold usually crosses zero rollovers. Direction matters: a long
    position is charged `long_swap_cost_pct_per_night`, a short position
    `short_swap_cost_pct_per_night` -- these are genuinely different,
    independently-signed real-world rates (see forex_ml.data.swap_rates), not one
    rate mirrored with a flip, so passing the same value for both would silently
    misprice one side. `flatten_before_rollover=True` implements the "flatten by
    5pm" rule instead of paying swap: any row whose holding period would cross a
    rollover is forced flat (skipped) rather than held through it and charged --
    both swap-cost params and `flatten_before_rollover` require
    `entry_timestamp`/`long_exit_timestamp`/`short_exit_timestamp`.

    `position_size` (default: all ones) scales both P&L and cost proportionally,
    so a 0.3-size position produces 30% of a full-size position's P&L AND 30% of
    its cost, not a discounted cost at full P&L -- see
    `position_size_from_realized_volatility` for a volatility-gated source.

    `long_raw_return_pct`/`short_raw_return_pct` are forex-ML's
    `Splits.test["long_raw_return_pct"]`/`["short_raw_return_pct"]` (or the
    predictions artifact's `test_long_raw_return_pct`/`test_short_raw_return_pct`)
    -- triple-barrier's per-side *pre-cost* realized % move at that side's own
    actual exit bar. Deliberately not `net_return_pct` (already net of spread/
    swap), which would double-count cost against the spread/swap this function
    charges itself. `spread`/`price` are the raw (non-percentage) price-unit
    values from `Splits.test`.
    """
    n = len(positions)
    if not (len(long_raw_return_pct) == len(short_raw_return_pct) == len(spread) == len(price) == n):
        raise ValueError(
            "positions/long_raw_return_pct/short_raw_return_pct/spread/price must all be the same length"
        )

    if position_size is None:
        position_size = np.ones(n)
    elif len(position_size) != n:
        raise ValueError("position_size must be the same length as positions")

    needs_timestamps = long_swap_cost_pct_per_night != 0.0 or short_swap_cost_pct_per_night != 0.0 \
        or flatten_before_rollover
    if needs_timestamps and (entry_timestamp is None or long_exit_timestamp is None or short_exit_timestamp is None):
        raise ValueError(
            "entry_timestamp, long_exit_timestamp, and short_exit_timestamp are required when "
            "long_swap_cost_pct_per_night or short_swap_cost_pct_per_night is set or "
            "flatten_before_rollover is True"
        )
    if needs_timestamps and not (
        len(entry_timestamp) == len(long_exit_timestamp) == len(short_exit_timestamp) == n  # type: ignore[arg-type]
    ):
        raise ValueError("entry_timestamp/long_exit_timestamp/short_exit_timestamp must be the same length as positions")

    raw_return_pct = np.where(positions > 0, long_raw_return_pct, short_raw_return_pct)

    if needs_timestamps:
        exit_timestamp = np.where(positions > 0, long_exit_timestamp, short_exit_timestamp)  # type: ignore[arg-type]
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
    gross_pnl_pct = sized_positions * raw_return_pct
    spread_cost_pct = np.where(is_trade, abs_size * 100.0 * spread / price, 0.0)
    swap_rate_per_row = np.where(positions > 0, long_swap_cost_pct_per_night, short_swap_cost_pct_per_night)
    swap_cost_pct = np.where(is_trade, abs_size * swap_rate_per_row * n_rollovers, 0.0)
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
