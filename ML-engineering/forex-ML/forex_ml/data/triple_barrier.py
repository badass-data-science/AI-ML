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

Bidirectional: two independent first-passage "races" are run per entry row, one
assuming a long position and one assuming a short, each against its own
swap-cost input (`long_swap_cost_pct_per_night`/`short_swap_cost_pct_per_night`
-- see forex_ml.data.swap_rates.resolve_swap_cost_pct_per_night, which already
resolves both sides from OANDA's live financing rates). Long's race is exactly
the original algorithm; short's race mirrors it with the position's own return
sign-flipped (`-raw_return_pct`), since a short profits when price falls. The
two races are NOT mirror images of each other once cost is included: long's
stop-loss fires as soon as a price drop breaches `-stop_loss_pct` net of cost
(cost adds to the loss, so a smaller drop triggers it), while a short's
profit-take requires the drop to clear `profit_take_pct` net of the short's OWN
cost (cost eats into the gain, so it takes a larger drop to be genuinely
profitable). Treating "long's stop-loss fired" as "short signal" -- the
original, pre-bidirectional design -- systematically overstated how easy it is
for a short to win; that's the bug this design fixes.

Merge rule: if long's race hits its profit-take (and short's doesn't, or hits
later) -> label +1. If short's race hits its profit-take (and long's doesn't,
or hits later) -> label -1. If neither race's profit-take fires (either side
timed out or hit its own stop-loss) -> label 0 (flat) -- expect this class to
grow relative to the original design, since a "short" label now requires
independently confirmed short profitability, not just "long lost." Ties
(both races' profit-takes firing at the exact same bar) are resolved with an
explicit rule (long wins) rather than assumed to be unreachable: the algebra
(`long_net + short_net = -2*entry_cost_pct - (long_swap + short_swap)`) shows a
same-bar double-fire is possible if entry_cost_pct goes negative or swap rates
are large enough credits, so `spread` is validated non-negative below to keep
entry_cost_pct itself non-negative, and the tie-break is real code, not an
unenforced invariant. For flat rows, the reported exit_bar_offset/net_return_pct
reference whichever race resolved first (tie -> long) -- the more truthful single
reference, since `forex_strategy.run_backtest` derives rollover-crossing counts
directly from exit_bar_offset for every test row.

Both races' own outcomes are ALSO persisted independently (long_exit_bar_offset/
long_raw_return_pct, short_exit_bar_offset/short_raw_return_pct -- see
TripleBarrierLabels), not just the merged single winning-side view above. This
matters for backtesting: a model's prediction can disagree with the label (that's
what "wrong" means), and a backtest evaluating that wrong-direction trade needs the
true outcome of the side actually taken, not the winning side's outcome as a
stand-in -- which is what `forex_strategy.backtest.simulate_trades` did before
these fields existed, mispricing every wrong-direction trade (see git history/
README for the specific fix and how large the effect was measured to be).

This IS the production training target -- `TimeSeriesSplitter`
(forex_ml/data/splitting.py) calls `triple_barrier_labels_from_frame` once per pair
in `__init__`, before any train/val/test splitting happens. `pd_lead`/
`spread_close_lead`/`volatility_lead` (forex_ml/data/features.py's `add_targets`)
are still computed in Stage 1 but are diagnostic-only reference columns now, not
selectable training targets -- see the README's "The prediction target:
triple-barrier labeling" section.
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
    label: np.ndarray            # +1 long's profit-take won, -1 short's profit-take won, 0 flat
    exit_bar_offset: np.ndarray  # bars from entry to the winning (or, if flat, earlier-resolving) race's exit
    net_return_pct: np.ndarray   # that race's realized % return at its exit bar, net of cost
    raw_return_pct: np.ndarray   # realized % return at that same exit bar, BEFORE cost -- what a
                                 # downstream backtest (which charges its own cost) should use as
                                 # "the move," rather than net_return_pct which would double-count it
    # long_*/short_* below are EACH SIDE'S OWN outcome, always both present regardless
    # of which side won -- unlike exit_bar_offset/raw_return_pct above, which only
    # reflect whichever race won (or resolved first, if flat). A downstream backtest
    # evaluating a model's prediction needs the outcome of the side actually taken,
    # not just whichever side the label happened to be. When a model predicts the
    # SAME direction as the label, long_raw_return_pct (or short_) exactly equals
    # raw_return_pct above; when it predicts the OPPOSITE direction, these are the
    # only correct source for that trade's true P&L -- substituting the winning
    # side's raw_return_pct instead (as forex_strategy.backtest.simulate_trades used
    # to do, before these fields existed) mispriced every wrong-direction trade,
    # roughly half of all trades in a near-50%-win-rate backtest.
    long_exit_bar_offset: np.ndarray
    long_raw_return_pct: np.ndarray
    short_exit_bar_offset: np.ndarray
    short_raw_return_pct: np.ndarray


def _run_single_side_race(
    price: np.ndarray,
    timestamp: np.ndarray,
    i: int,
    entry_price: float,
    entry_ts: float,
    entry_cost_pct: float,
    profit_take_pct: float,
    stop_loss_pct: float,
    max_holding_bars: int,
    swap_cost_pct_per_night: float,
    sign: float,
) -> tuple[int, int, float, float]:
    """Walk bars 1..max_holding_bars forward from entry bar `i`, checking one
    side's own economics (`sign=+1` long, `sign=-1` short) against the barriers.
    Returns (hit_type, exit_bar_offset, net_return_pct, raw_return_pct), where
    hit_type is +1 (this side's profit-take fired), -1 (this side's stop-loss
    fired), or 0 (timed out)."""
    for j in range(1, max_holding_bars + 1):
        exit_ts = timestamp[i + j]
        raw_return_pct = 100.0 * (price[i + j] - entry_price) / entry_price
        swap_cost_pct = swap_cost_pct_per_night * count_rollovers_crossed(entry_ts, exit_ts)
        net = sign * raw_return_pct - entry_cost_pct - swap_cost_pct

        if net >= profit_take_pct:
            return 1, j, net, raw_return_pct
        if net <= -stop_loss_pct:
            return -1, j, net, raw_return_pct

    j = max_holding_bars
    raw_return_pct = 100.0 * (price[i + j] - entry_price) / entry_price
    swap_cost_pct = swap_cost_pct_per_night * count_rollovers_crossed(entry_ts, timestamp[i + j])
    net = sign * raw_return_pct - entry_cost_pct - swap_cost_pct
    return 0, j, net, raw_return_pct


def triple_barrier_labels(
    price: np.ndarray,
    spread: np.ndarray,
    timestamp: np.ndarray,
    profit_take_pct: float,
    stop_loss_pct: float,
    max_holding_bars: int,
    long_swap_cost_pct_per_night: float = 0.0,
    short_swap_cost_pct_per_night: float = 0.0,
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
    if np.any(spread < 0):
        raise ValueError("spread must be non-negative")

    n_labelable = n - max_holding_bars
    label = np.zeros(n_labelable, dtype=int)
    exit_bar_offset = np.full(n_labelable, max_holding_bars, dtype=int)
    net_return_pct = np.zeros(n_labelable, dtype=float)
    raw_return_pct_out = np.zeros(n_labelable, dtype=float)
    long_exit_bar_offset = np.full(n_labelable, max_holding_bars, dtype=int)
    long_raw_return_pct = np.zeros(n_labelable, dtype=float)
    short_exit_bar_offset = np.full(n_labelable, max_holding_bars, dtype=int)
    short_raw_return_pct = np.zeros(n_labelable, dtype=float)

    for i in range(n_labelable):
        entry_price = price[i]
        entry_ts = timestamp[i]
        entry_cost_pct = 100.0 * spread[i] / entry_price  # one round-trip spread charge

        long_hit, long_j, long_net, long_raw = _run_single_side_race(
            price, timestamp, i, entry_price, entry_ts, entry_cost_pct,
            profit_take_pct, stop_loss_pct, max_holding_bars,
            long_swap_cost_pct_per_night, sign=1.0,
        )
        short_hit, short_j, short_net, short_raw = _run_single_side_race(
            price, timestamp, i, entry_price, entry_ts, entry_cost_pct,
            profit_take_pct, stop_loss_pct, max_holding_bars,
            short_swap_cost_pct_per_night, sign=-1.0,
        )

        long_exit_bar_offset[i], long_raw_return_pct[i] = long_j, long_raw
        short_exit_bar_offset[i], short_raw_return_pct[i] = short_j, short_raw

        long_wins = long_hit == 1
        short_wins = short_hit == 1

        if long_wins and (not short_wins or long_j <= short_j):
            label[i], exit_bar_offset[i], net_return_pct[i] = 1, long_j, long_net
            raw_return_pct_out[i] = long_raw
        elif short_wins:
            label[i], exit_bar_offset[i], net_return_pct[i] = -1, short_j, short_net
            raw_return_pct_out[i] = short_raw
        else:
            # Flat: neither side's profit-take fired. Report whichever race
            # resolved first (tie -> long) as the single reference -- that's the
            # race whose outcome is actually known soonest, not an arbitrary pick.
            if long_j <= short_j:
                exit_bar_offset[i], net_return_pct[i], raw_return_pct_out[i] = long_j, long_net, long_raw
            else:
                exit_bar_offset[i], net_return_pct[i], raw_return_pct_out[i] = short_j, short_net, short_raw

    return TripleBarrierLabels(
        label=label, exit_bar_offset=exit_bar_offset,
        net_return_pct=net_return_pct, raw_return_pct=raw_return_pct_out,
        long_exit_bar_offset=long_exit_bar_offset, long_raw_return_pct=long_raw_return_pct,
        short_exit_bar_offset=short_exit_bar_offset, short_raw_return_pct=short_raw_return_pct,
    )


def triple_barrier_labels_from_frame(
    df: pd.DataFrame,
    profit_take_pct: float,
    stop_loss_pct: float,
    max_holding_bars: int,
    long_swap_cost_pct_per_night: float = 0.0,
    short_swap_cost_pct_per_night: float = 0.0,
    price_column: str = "mid_close",
    spread_column: str = "spread_close",
    timestamp_column: str = "unix_epoch_s",
) -> pd.DataFrame:
    """Convenience wrapper for a Stage-1-shaped DataFrame (must already be sorted
    chronologically for one (instrument, granularity) pair -- e.g. forex-ML's
    `df_non_time_series` Parquet output, which carries mid_close/spread_close as
    passthrough reference columns; see COLUMNS_PASSTHROUGH in features.py).
    Returns `df`'s first `len(df) - max_holding_bars` rows with label/
    exit_bar_offset/net_return_pct/raw_return_pct/long_exit_bar_offset/
    long_raw_return_pct/short_exit_bar_offset/short_raw_return_pct columns
    appended (see TripleBarrierLabels for what the long_*/short_* columns are)."""
    result = triple_barrier_labels(
        df[price_column].to_numpy(),
        df[spread_column].to_numpy(),
        df[timestamp_column].to_numpy(),
        profit_take_pct, stop_loss_pct, max_holding_bars,
        long_swap_cost_pct_per_night, short_swap_cost_pct_per_night,
    )
    n_labelable = len(df) - max_holding_bars
    out = df.iloc[:n_labelable].copy()
    out["label"] = result.label
    out["exit_bar_offset"] = result.exit_bar_offset
    out["net_return_pct"] = result.net_return_pct
    out["raw_return_pct"] = result.raw_return_pct
    out["long_exit_bar_offset"] = result.long_exit_bar_offset
    out["long_raw_return_pct"] = result.long_raw_return_pct
    out["short_exit_bar_offset"] = result.short_exit_bar_offset
    out["short_raw_return_pct"] = result.short_raw_return_pct
    return out
