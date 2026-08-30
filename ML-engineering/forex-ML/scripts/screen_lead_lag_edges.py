"""Screens forex-partial-correlation-network's (fx-pcn) FDR-corrected directed
edges for a tradeable lead-lag signal, using this project's own multi-threshold
BH-corrected verdict framework (forex_ml.evaluation.pair_screening) -- same
statistical discipline as scripts/screen_pair.py, applied to a completely
different signal source.

The question: on a date `t` where fx-pcn's rolling window fit found pair L
Granger-causes pair F (direction survives FDR correction, i.e. NOT
"undirected"), does L's next realized bar's return actually predict F's
return one bar later? An "i<->j" (bidirected) edge is treated as BOTH
directions being tradeable simultaneously.

Trade construction (all non-lookahead by construction; granularity-agnostic --
works the same whether a "bar" is an H1 candle or a full Daily candle):
1. leader_bar(k) = L's (k-1)-th bar after midnight of (date + 1 day), for each
   lag k in 1..max_lag (max_lag read from the edges file itself, e.g. 12 for
   the M15 intraday regime -- matching how many lags the Granger test itself
   checked jointly). k=1 is "the next calendar day after the fit window
   closes"; k=2..max_lag walk further into that day. This is a simplification
   of what Granger actually tested (a single joint regression on lags 1..k,
   not k independent single-lag checks) but is the natural way to ask
   "which lag depth, if any, actually carries tradeable signal" as a set of
   separate, honestly-reported trading rules.
2. leader_return_pct = simple close-to-close return over leader_bar(k), i.e.
   (leader_bar.mid_close - prior_bar.mid_close) / prior_bar.mid_close.
   (Stage 1's own `return` column is a length-200 lookback ARRAY built for
   model input, not a per-bar scalar -- deliberately not reused here.)
3. follower_bar = F's first bar strictly after leader_bar(k)'s own timestamp
   (immediately following bar -- realized strictly after the leader signal
   is known).
4. follower_return_pct computed the same way as step 2, on F's own series.
5. position sign = sign(partial_corr) * sign(leader_return_pct); skip if
   leader_return_pct == 0 (no signal at that lag).
6. Fed straight into trade_simulator.backtest.simulate_trades as a 1-bar-hold
   trade (long_raw_return_pct = follower_return_pct*100, short_raw_return_pct
   = its negation -- correct here, not the usual asymmetric-barrier case,
   because the exit rule is a fixed 1-bar hold, not a barrier hit, so long
   and short outcomes over the SAME bar are exact mirror images). No swap
   cost modeled (H1/M15 holds are ~minutes to hours, essentially never cross
   a rollover; a Daily-granularity regime's 1-bar hold DOES span a rollover
   and this is a known simplification for that case).

Every (edge, lag) combination is an independent trade opportunity -- the main
verdict pools ALL lags together per |partial_corr| threshold (more trades,
more power, but a real effect concentrated in one lag can get diluted by
noise at the others); a separate per-lag descriptive breakdown (no BH
correction, just win_rate/net_pnl) is also printed so a lag-specific pattern
isn't hidden by the pooled number.

Price data source depends on the edges file's own `granularity` column: H1
reuses forex-ML's existing Stage-1 parquet (fast, already on disk for every
major); anything else (D, M15, ...) pulls raw candles directly from InfluxDB
via forex_ml.data.influx_source.pull_candles (no forex-ML Stage-1 build exists
for a full 7-pair universe at other granularities) -- needs INFLUXDB_URL/
TOKEN/ORG/BUCKET set.

Reuses `min_abs_partial_corr` as the threshold axis (analogous to
screen_pair.py's confidence thresholds) and forex_ml.evaluation.pair_screening's
existing BH-corrected verdict logic UNCHANGED.

Critically -- learned the hard way earlier this session -- reports the
FULL-HISTORY, PRE-2023, and 2023+ periods SEPARATELY rather than one pooled
number, so a result that only holds in stale history can't hide behind a
pooled figure.

Usage (same venv convention as screen_pair.py -- needs trade_simulator; run
with cwd at forex-ML/ so the relative Stage-1 paths resolve):
    cd forex-ML
    uv run --project ../forex-strategy python scripts/screen_lead_lag_edges.py \\
        --edges /home/emily/output/forex-partial-correlation-network/parameters---network-name-forex-network-seven-majors---window-days-5---step-days-1---min-observations-60---max-lag-4---fdr-alpha-0.05---granularity-H1.parquet \\
        [--thresholds 0.0,0.05,0.10,0.15,0.20] [--recent-cutoff 2023-01-01] [--max-lag N] [--alpha 0.05]
"""

from __future__ import annotations

import argparse
import datetime
import glob

import numpy as np
import pandas as pd

from forex_ml.evaluation.pair_screening import evaluate_screening_verdict, pool_fold_results
from trade_simulator.backtest import simulate_trades

STAGE1_DIR = "output/interim"
DEFAULT_THRESHOLDS = [0.0, 0.05, 0.10, 0.15, 0.20]


def _load_price_series_h1(pair: str) -> pd.DataFrame:
    """H1 has a ready-made Stage-1 parquet for every major already on disk --
    reuse it (fast, no network) rather than re-pulling from InfluxDB."""
    key = pair.replace("/", "_")
    paths = sorted(glob.glob(f"{STAGE1_DIR}/df_time_series__{key}__H1__200__4.parquet/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Stage-1 H1 parquet found for {pair} under {STAGE1_DIR}")
    frames = [pd.read_parquet(p, columns=["unix_epoch_s", "mid_close", "spread_close"]) for p in paths]
    df = pd.concat(frames).drop_duplicates("unix_epoch_s").sort_values("unix_epoch_s").reset_index(drop=True)
    return df


def _load_price_series_influx(pair: str, granularity: str, min_ts: int, max_ts: int) -> pd.DataFrame:
    """Other granularities (D, M15, ...) have no ready-made forex-ML Stage-1
    parquet for a full 7-pair universe -- pull raw candles directly. No Spark
    needed, `pull_candles` is a plain pandas/Flux-query function."""
    from forex_ml.data.influx_source import pull_candles

    df = pull_candles(pair, granularity, min_ts, max_ts)
    return df[["unix_epoch_s", "mid_close", "spread_close"]].sort_values("unix_epoch_s").reset_index(drop=True)


def _bar_after(df: pd.DataFrame, ts: float) -> int | None:
    """Index of the first row with unix_epoch_s > ts, or None if there isn't one."""
    idx = df["unix_epoch_s"].searchsorted(ts, side="right")
    return int(idx) if idx < len(df) else None


def _pct_return_at(df: pd.DataFrame, idx: int) -> float | None:
    """Simple close-to-close return realized OVER this bar (prior bar's close to
    this bar's close) -- None if there's no prior bar."""
    if idx == 0:
        return None
    prior_close = df["mid_close"].iloc[idx - 1]
    if prior_close == 0:
        return None
    return float((df["mid_close"].iloc[idx] - prior_close) / prior_close)


def _leading_edges(edges: pd.DataFrame, direction_column: str = "direction") -> pd.DataFrame:
    """Expands direction-labeled edge rows into one row per (leader, follower).
    fx-pcn's direction columns are NOT the literal string "i->j" -- they
    substitute the real pair names, e.g. "USD/CHF->AUD/USD" or
    "EUR/USD<->GBP/USD" (see fx_pcn/network.py: f'{i}->{j}'/f'{i}<->{j}'/
    f'{j}->{i}'/'undirected'). `i->j`/`j->i` keep just one row; `i<->j` becomes
    two (both directions tradeable); "undirected"/"no_edge" rows are dropped
    (no lead-lag direction to trade). `direction_column` lets this same
    parsing logic serve both the steady-state edge table's `direction` column
    and the direction-flips table's `new_direction` column."""
    rows = []
    for row in edges.itertuples(index=False):
        direction = getattr(row, direction_column)
        if direction == f"{row.pair_i}->{row.pair_j}":
            rows.append((row.date, row.pair_i, row.pair_j, row.partial_corr))
        elif direction == f"{row.pair_j}->{row.pair_i}":
            rows.append((row.date, row.pair_j, row.pair_i, row.partial_corr))
        elif direction == f"{row.pair_i}<->{row.pair_j}":
            rows.append((row.date, row.pair_i, row.pair_j, row.partial_corr))
            rows.append((row.date, row.pair_j, row.pair_i, row.partial_corr))
        # "undirected"/"no_edge": no direction to trade, skip.
    return pd.DataFrame(rows, columns=["date", "leader", "follower", "partial_corr"])


def load_flip_edges(flips_path: str, edges: pd.DataFrame) -> pd.DataFrame:
    """Loads fx-pcn's direction-FLIPS table (one row per date a specific pair's
    relationship just changed) and joins back to the steady-state edge table to
    recover `partial_corr` (the flips table itself doesn't carry it). Restricts
    to flips whose `new_direction` is an actual directed edge -- a flip INTO
    "undirected" or "no_edge" has no direction left to trade. `pair_i`/`pair_j`
    ordering is guaranteed identical between the two tables (fx_pcn.
    direction_flips.find_direction_flips merges directly against the edge
    table's own pair_i/pair_j, never re-derives or re-sorts them), so the join
    key is exact."""
    flips = pd.read_parquet(flips_path)
    flips = flips[~flips["new_direction"].isin(["undirected", "no_edge"])].reset_index(drop=True)
    joined = flips.merge(
        edges[["date", "pair_i", "pair_j", "partial_corr"]],
        on=["date", "pair_i", "pair_j"], how="inner",
    )
    return joined


def build_trades(
    leading: pd.DataFrame, granularity: str, min_ts: int, max_ts: int, max_lag: int,
) -> pd.DataFrame:
    """One row per tradeable (date, leader, follower, lag) edge with a real,
    computed leader_return_pct/follower_return_pct -- edges where either leg
    falls off the end of either series (not enough trailing data yet, or the
    requested lag walks past the end of the day's/series' data) are silently
    dropped, same as any other walk-forward backtest boundary effect.
    `leading` is a (date, leader, follower, partial_corr) table -- from either
    `_leading_edges(edges)` (steady-state: trade every day an edge holds) or
    `_leading_edges(load_flip_edges(...), direction_column="new_direction")`
    (flip-gated: trade only the day a directed edge first appears/reverses)."""
    price_cache: dict[str, pd.DataFrame] = {}

    if granularity == "H1":
        def get_price(pair: str) -> pd.DataFrame:
            if pair not in price_cache:
                price_cache[pair] = _load_price_series_h1(pair)
            return price_cache[pair]
    else:
        def get_price(pair: str) -> pd.DataFrame:
            if pair not in price_cache:
                price_cache[pair] = _load_price_series_influx(pair, granularity, min_ts, max_ts)
            return price_cache[pair]

    records = []
    for row in leading.itertuples(index=False):
        leader_df = get_price(row.leader)
        follower_df = get_price(row.follower)

        next_day_start = datetime.datetime.combine(
            row.date + datetime.timedelta(days=1), datetime.time.min
        ).timestamp()
        day_start_idx = _bar_after(leader_df, next_day_start - 1)  # first bar AT or after next-day start
        if day_start_idx is None:
            continue

        for lag in range(1, max_lag + 1):
            leader_idx = day_start_idx + (lag - 1)
            if leader_idx >= len(leader_df):
                break  # walked past the end of the series; higher lags won't fare better
            leader_return = _pct_return_at(leader_df, leader_idx)
            if leader_return is None or leader_return == 0:
                continue

            leader_ts = float(leader_df["unix_epoch_s"].iloc[leader_idx])
            follower_idx = _bar_after(follower_df, leader_ts)
            if follower_idx is None:
                continue
            follower_return = _pct_return_at(follower_df, follower_idx)
            if follower_return is None:
                continue

            records.append({
                "date": row.date,
                "leader": row.leader,
                "follower": row.follower,
                "lag": lag,
                "partial_corr": row.partial_corr,
                "leader_return_pct": leader_return,
                "follower_return_pct": follower_return,
                "follower_spread": float(follower_df["spread_close"].iloc[follower_idx]),
                "follower_price": float(follower_df["mid_close"].iloc[follower_idx - 1]),
            })
    columns = ["date", "leader", "follower", "lag", "partial_corr", "leader_return_pct",
               "follower_return_pct", "follower_spread", "follower_price"]
    return pd.DataFrame.from_records(records, columns=columns)


def simulate_threshold(trades: pd.DataFrame, min_abs_partial_corr: float):
    subset = trades[trades["partial_corr"].abs() >= min_abs_partial_corr]
    if subset.empty:
        return None
    positions = np.sign(subset["partial_corr"].to_numpy() * subset["leader_return_pct"].to_numpy())
    long_raw_return_pct = subset["follower_return_pct"].to_numpy() * 100.0
    short_raw_return_pct = -long_raw_return_pct
    return simulate_trades(
        positions, long_raw_return_pct, short_raw_return_pct,
        subset["follower_spread"].to_numpy(), subset["follower_price"].to_numpy(),
    )


def report_period(label: str, trades: pd.DataFrame, thresholds: list[float], alpha: float) -> None:
    print(f"\n{'=' * 70}\n{label}  (n_edges_available={len(trades)})\n{'=' * 70}", flush=True)
    if trades.empty:
        print("  no trades in this period", flush=True)
        return
    pooled = {}
    for thr in thresholds:
        result = simulate_threshold(trades, thr)
        if result is None or result.n_trades == 0:
            continue
        pooled[thr] = pool_fold_results(thr, [result])
    if not pooled:
        print("  no threshold produced any trades", flush=True)
        return
    for thr, r in sorted(pooled.items()):
        print(f"  |rho|>={thr:.2f}  trades={r.total_trades:5d}  win_rate={r.pooled_win_rate:.3f}  "
              f"win_rate_p={r.win_rate_p_value:.4f}  net_pnl_pct={r.total_net_pnl_pct:8.3f}  "
              f"per_trade_t_p={r.per_trade_t_p_value:.4f}  per_trade_w_p={r.per_trade_wilcoxon_p_value:.4f}",
              flush=True)
    verdict = evaluate_screening_verdict(pooled, alpha=alpha)
    print(f"\n  VERDICT: {'PASS' if verdict.passed else 'FAIL'} -- {verdict.reason}", flush=True)
    if verdict.passed:
        print(f"  Passing thresholds: {verdict.passing_thresholds}", flush=True)

    print("\n  Per-lag breakdown (all thresholds pooled, descriptive only -- NOT BH-corrected):", flush=True)
    for lag, lag_trades in trades.groupby("lag"):
        result = simulate_threshold(lag_trades, min_abs_partial_corr=0.0)
        if result is None or result.n_trades == 0:
            continue
        print(f"    lag={lag:2d}  trades={result.n_trades:5d}  win_rate={result.win_rate:.3f}  "
              f"net_pnl_pct={result.net_pnl_pct:8.3f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--edges", required=True, help="Path to an fx-pcn edge-table ('parameters---...') parquet")
    parser.add_argument("--flips", default=None,
                         help="Path to an fx-pcn direction-flips table ('direction---...') parquet. If given, "
                              "trades ONLY the day a directed edge first appears/reverses (flip-gated), instead "
                              "of every day it happens to hold (steady-state, the default).")
    parser.add_argument("--thresholds", default=",".join(str(t) for t in DEFAULT_THRESHOLDS),
                         help="Comma-separated min |partial_corr| thresholds")
    parser.add_argument("--recent-cutoff", default="2023-01-01", help="ISO date splitting historical vs recent")
    parser.add_argument("--max-lag", type=int, default=None,
                         help="Test leader lags 1..N as separate trade opportunities (see module docstring). "
                              "Default: the edges file's own max_lag column.")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    thresholds = [float(t) for t in args.thresholds.split(",")]
    recent_cutoff = datetime.date.fromisoformat(args.recent_cutoff)

    edges = pd.read_parquet(args.edges)
    edges = edges[edges["direction"] != "undirected"].reset_index(drop=True)
    print(f"Loaded {len(edges)} directed edge-rows from {args.edges}", flush=True)

    if args.flips is not None:
        flip_joined = load_flip_edges(args.flips, edges)
        print(f"Loaded {len(flip_joined)} flip-into-a-real-direction events from {args.flips}", flush=True)
        leading = _leading_edges(flip_joined, direction_column="new_direction")
    else:
        leading = _leading_edges(edges)

    max_lag = args.max_lag if args.max_lag is not None else int(edges["max_lag"].iloc[0])
    print(f"Testing lags 1..{max_lag}", flush=True)

    granularity = str(edges["granularity"].iloc[0])
    min_ts = int(datetime.datetime.combine(edges["date"].min(), datetime.time.min).timestamp())
    max_ts = int(datetime.datetime.combine(
        edges["date"].max() + datetime.timedelta(days=int(edges["window_days"].iloc[0]) + 5), datetime.time.min,
    ).timestamp())

    trades = build_trades(leading, granularity=granularity, min_ts=min_ts, max_ts=max_ts, max_lag=max_lag)
    print(f"Built {len(trades)} tradeable (leader, follower, lag) instances", flush=True)

    report_period("FULL HISTORY", trades, thresholds, args.alpha)
    report_period(f"HISTORICAL (before {recent_cutoff})", trades[trades["date"] < recent_cutoff], thresholds, args.alpha)
    report_period(f"RECENT ({recent_cutoff} onward)", trades[trades["date"] >= recent_cutoff], thresholds, args.alpha)

    print("\nALL_DONE", flush=True)


if __name__ == "__main__":
    main()
