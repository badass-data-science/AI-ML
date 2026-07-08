"""Triple-barrier labeling (Lopez de Prado, *Advances in Financial Machine
Learning*): label each candidate entry by whichever of three barriers is hit
first -- an upper (profit-take) barrier, a lower (stop-loss) barrier, or a
vertical (max-holding-period) barrier -- rather than a fixed-horizon percent
change like `pd_lead`. This is a genuinely different, event-driven notion of
"the label": it matches how a real trade actually closes (hits its target, hits
its stop, or times out), rather than always measuring the move over a fixed
number of bars regardless of what happened in between.

Cost-aware: profit_take_pct/stop_loss_pct are thresholds on the NET (cost-
adjusted) return, not the raw price move -- so a "profit" label means the trade
would have cleared costs, not just that price moved the right way. Spread is
charged once, as a full round-trip cost (same convention as
`forex_strategy.backtest`); swap/rollover is charged once per 5pm New York
rollover boundary actually crossed between entry and exit, not per bar held,
since that's how OANDA actually charges it -- an intraday (H1/M15) hold crosses
zero rollovers most of the time, and multi-day holds accumulate one charge per
night, not one per bar.

Long-side labeling only: the upper barrier is a profit-take and the lower a
stop-loss FOR A LONG position. `swap_cost_pct_per_night` should be supplied by
the caller as whatever a long position actually gets charged (e.g. the negative
of OANDA's `long_rate` financing rate from forex-etl's SwapRateRecord, if
`long_rate` is negative -- a positive `long_rate` is a credit, not a cost).

This is a standalone labeling/research utility, not yet wired into Stage 1's
`add_targets`/`column_y` -- replacing the pipeline's production target is a
bigger, separate decision (retraining, re-validating baselines, etc.) than
building and testing the labeling method itself.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

_NY_TZ = ZoneInfo("America/New_York")
_ROLLOVER_HOUR_NY = 17


def count_rollovers_crossed(entry_ts: float, exit_ts: float) -> int:
    """Number of 5pm America/New_York rollover boundaries strictly after entry_ts
    and at-or-before exit_ts -- i.e. how many nights this holding period would
    actually be charged swap for. DST-aware (unlike the rest of this pipeline's
    fixed-UTC trading-session-window simplification, see features.py) -- an
    hour's error here is the difference between being charged a night's swap or
    not, not just a soft diurnal-pattern approximation."""
    entry_dt = datetime.datetime.fromtimestamp(entry_ts, tz=_NY_TZ)
    exit_dt = datetime.datetime.fromtimestamp(exit_ts, tz=_NY_TZ)

    candidate = entry_dt.replace(hour=_ROLLOVER_HOUR_NY, minute=0, second=0, microsecond=0)
    if candidate <= entry_dt:
        candidate += datetime.timedelta(days=1)

    count = 0
    while candidate <= exit_dt:
        count += 1
        candidate += datetime.timedelta(days=1)
    return count


@dataclass
class TripleBarrierLabels:
    label: np.ndarray            # +1 profit-take hit, -1 stop-loss hit, 0 timed out
    exit_bar_offset: np.ndarray  # bars from entry to the exit (max_holding_bars if timed out)
    net_return_pct: np.ndarray   # realized % return at the exit bar, net of cost


def triple_barrier_labels(
    price: np.ndarray,
    spread: np.ndarray,
    timestamp: np.ndarray,
    profit_take_pct: float,
    stop_loss_pct: float,
    max_holding_bars: int,
    swap_cost_pct_per_night: float = 0.0,
) -> TripleBarrierLabels:
    """`price`/`spread`/`timestamp` must already be sorted chronologically for one
    (instrument, granularity) pair (e.g. mid_close/spread_close/unix_epoch_s from
    forex-ML's Stage 1 output). Returns arrays of length
    `len(price) - max_holding_bars` -- the last `max_holding_bars` rows don't have
    a full forward window to check barriers against, the same reason `pd_lead`
    is NaN for the final `lookahead` rows.
    """
    n = len(price)
    if not (len(spread) == len(timestamp) == n):
        raise ValueError("price/spread/timestamp must all be the same length")
    if profit_take_pct <= 0 or stop_loss_pct <= 0:
        raise ValueError("profit_take_pct and stop_loss_pct must be positive")
    if max_holding_bars < 1:
        raise ValueError("max_holding_bars must be >= 1")
    if n <= max_holding_bars:
        raise ValueError(f"Need more than max_holding_bars={max_holding_bars} rows, got {n}")

    n_labelable = n - max_holding_bars
    label = np.zeros(n_labelable, dtype=int)
    exit_bar_offset = np.full(n_labelable, max_holding_bars, dtype=int)
    net_return_pct = np.zeros(n_labelable, dtype=float)

    for i in range(n_labelable):
        entry_price = price[i]
        entry_ts = timestamp[i]
        entry_cost_pct = 100.0 * spread[i] / entry_price  # one round-trip spread charge

        hit = False
        for j in range(1, max_holding_bars + 1):
            exit_ts = timestamp[i + j]
            raw_return_pct = 100.0 * (price[i + j] - entry_price) / entry_price
            swap_cost_pct = swap_cost_pct_per_night * count_rollovers_crossed(entry_ts, exit_ts)
            net = raw_return_pct - entry_cost_pct - swap_cost_pct

            if net >= profit_take_pct:
                label[i], exit_bar_offset[i], net_return_pct[i] = 1, j, net
                hit = True
                break
            if net <= -stop_loss_pct:
                label[i], exit_bar_offset[i], net_return_pct[i] = -1, j, net
                hit = True
                break

        if not hit:
            j = max_holding_bars
            raw_return_pct = 100.0 * (price[i + j] - entry_price) / entry_price
            swap_cost_pct = swap_cost_pct_per_night * count_rollovers_crossed(entry_ts, timestamp[i + j])
            net_return_pct[i] = raw_return_pct - entry_cost_pct - swap_cost_pct

    return TripleBarrierLabels(label=label, exit_bar_offset=exit_bar_offset, net_return_pct=net_return_pct)


def triple_barrier_labels_from_frame(
    df: pd.DataFrame,
    profit_take_pct: float,
    stop_loss_pct: float,
    max_holding_bars: int,
    swap_cost_pct_per_night: float = 0.0,
    price_column: str = "mid_close",
    spread_column: str = "spread_close",
    timestamp_column: str = "unix_epoch_s",
) -> pd.DataFrame:
    """Convenience wrapper for a Stage-1-shaped DataFrame (must already be sorted
    chronologically for one (instrument, granularity) pair -- e.g. forex-ML's
    `df_non_time_series` Parquet output, which carries mid_close/spread_close as
    passthrough reference columns; see COLUMNS_PASSTHROUGH in features.py).
    Returns `df`'s first `len(df) - max_holding_bars` rows with label/
    exit_bar_offset/net_return_pct columns appended."""
    result = triple_barrier_labels(
        df[price_column].to_numpy(),
        df[spread_column].to_numpy(),
        df[timestamp_column].to_numpy(),
        profit_take_pct, stop_loss_pct, max_holding_bars, swap_cost_pct_per_night,
    )
    n_labelable = len(df) - max_holding_bars
    out = df.iloc[:n_labelable].copy()
    out["label"] = result.label
    out["exit_bar_offset"] = result.exit_bar_offset
    out["net_return_pct"] = result.net_return_pct
    return out
