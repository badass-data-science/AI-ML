"""Ad hoc validation (not a committed tool): does fx-pcn's network DENSITY
carry real information about upcoming market stress, independent of any
directional trading signal?

Two checks, run against ALL FOUR fx-pcn regimes (default/H1, intraday/M15,
macro/D-60, policy/D-180) -- density.parquet doesn't carry a granularity or
step_days its forward-return construction needs to change for, since it's
already one row per actual fit DATE regardless of underlying bar granularity;
only each regime's own approximate lookback (window_days) differs, used for
check 1's trailing-correlation comparison window:
1. Sanity check: does fx-pcn's own mean_abs_partial_corr track a plain,
   independently-computed realized trailing correlation among the 7 majors'
   daily returns? (confirms the density measure means what we think it means)
2. The actual premise: does elevated density/mean_abs_partial_corr on date t
   predict elevated realized volatility (or a larger drawdown) for a naive
   equal-weighted 7-pair basket over the FOLLOWING N days -- i.e. is it a
   genuine LEADING indicator, not just a contemporaneous label -- tested on
   full history AND recent-only (2023+), same recency discipline as every
   other test this session.

Run from forex-ML/ with forex-strategy's venv:
    uv run --project ../forex-strategy python /path/to/this_file.py
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
from scipy import stats

from forex_ml.data.influx_source import pull_candles

PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "USD/CAD", "AUD/USD", "NZD/USD"]
OUTPUT_DIR = "/home/emily/output/forex-partial-correlation-network"
# density.parquet doesn't carry its own run params, so window_days is repeated here
# per regime (used only for check 1's trailing-correlation window, to roughly match
# each regime's own lookback).
REGIMES = {
    "default (H1)": {
        "path": f"{OUTPUT_DIR}/density---network-name-forex-network-seven-majors---window-days-5---"
                f"step-days-1---min-observations-60---max-lag-4---fdr-alpha-0.05---granularity-H1.parquet",
        "window_days": 5,
    },
    "intraday (M15)": {
        "path": f"{OUTPUT_DIR}/density---network-name-forex-network-seven-majors---window-days-2---"
                f"step-days-1---min-observations-96---max-lag-12---fdr-alpha-0.05---granularity-M15.parquet",
        "window_days": 2,
    },
    "macro (D-60)": {
        "path": f"{OUTPUT_DIR}/density---network-name-forex-network-seven-majors---window-days-60---"
                f"step-days-7---min-observations-30---max-lag-3---fdr-alpha-0.05---granularity-D.parquet",
        "window_days": 60,
    },
    "policy (D-180)": {
        "path": f"{OUTPUT_DIR}/density---network-name-forex-network-seven-majors---window-days-180---"
                f"step-days-30---min-observations-90---max-lag-7---fdr-alpha-0.05---granularity-D.parquet",
        "window_days": 180,
    },
}
FORWARD_WINDOW_DAYS = 5
RECENT_CUTOFF = datetime.date(2023, 1, 1)


def load_daily_returns() -> pd.DataFrame:
    min_ts = int(datetime.datetime(2015, 1, 1).timestamp())
    max_ts = int(datetime.datetime.now().timestamp())
    returns = {}
    for pair in PAIRS:
        df = pull_candles(pair, "D", min_ts, max_ts).sort_values("unix_epoch_s")
        df["date"] = pd.to_datetime(df["unix_epoch_s"], unit="s").dt.date
        df = df.drop_duplicates("date").set_index("date")
        returns[pair] = df["mid_close"].pct_change()
    return pd.DataFrame(returns).sort_index()


def report(label: str, df: pd.DataFrame) -> None:
    print(f"\n  --- {label}, n={len(df)} ---")
    if len(df) < 20:
        print("    too few rows to report meaningfully, skipping")
        return
    for col in ("density", "mean_abs_partial_corr"):
        rho_v, p_v = stats.spearmanr(df[col], df["fwd_vol"])
        rho_d, p_d = stats.spearmanr(df[col], df["fwd_worst_day"])
        print(f"    {col} vs fwd_{FORWARD_WINDOW_DAYS}d_vol:        rho={rho_v:+.3f}  p={p_v:.4f}")
        print(f"    {col} vs fwd_{FORWARD_WINDOW_DAYS}d_worst_day:  rho={rho_d:+.3f}  p={p_d:.4f}")
    for col in ("mean_abs_partial_corr", "density"):
        try:
            quartiles = pd.qcut(df[col], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop")
        except ValueError:
            print(f"    (too few distinct {col} values for quartiles)")
            continue
        summary = df.groupby(quartiles, observed=True)[["fwd_vol", "fwd_worst_day"]].agg(["mean", "count"])
        print(f"\n    By {col} quartile (mean forward {FORWARD_WINDOW_DAYS}d vol / worst single day, count):")
        print(summary.to_string().replace("\n", "\n    "))


def main() -> None:
    returns = load_daily_returns()
    basket_ret = returns.mean(axis=1)  # naive equal-weighted average daily return

    # Forward-looking targets are regime-independent (same basket, same forward
    # window) -- computed once, then joined against each regime's own density dates.
    fwd_vol = basket_ret.iloc[::-1].rolling(FORWARD_WINDOW_DAYS).std().iloc[::-1].shift(-1)
    fwd_maxdd = -basket_ret.iloc[::-1].rolling(FORWARD_WINDOW_DAYS).min().iloc[::-1].shift(-1)

    for regime_label, regime in REGIMES.items():
        print(f"\n{'#' * 70}\n# REGIME: {regime_label}\n{'#' * 70}")
        density = pd.read_parquet(regime["path"]).set_index("date").sort_index()

        # --- Check 1: does this regime's mean_abs_partial_corr track a plain
        # realized trailing correlation among the 7 pairs' own daily returns,
        # using a comparably-sized trailing window? ---
        trailing_days = regime["window_days"]
        trailing_corr = pd.Series(index=returns.index, dtype=float)
        for i in range(trailing_days, len(returns)):
            window = returns.iloc[i - trailing_days:i]
            corr_matrix = window.corr().to_numpy()
            n = corr_matrix.shape[0]
            off_diag = corr_matrix[np.triu_indices(n, k=1)]
            trailing_corr.iloc[i] = np.nanmean(np.abs(off_diag))

        sanity = pd.DataFrame({
            "fx_pcn_mean_abs_partial_corr": density["mean_abs_partial_corr"],
            "realized_trailing_abs_corr": trailing_corr,
        }).dropna()
        if len(sanity) >= 20:
            rho, p = stats.spearmanr(sanity["fx_pcn_mean_abs_partial_corr"], sanity["realized_trailing_abs_corr"])
            print(f"CHECK 1 (does mean_abs_partial_corr track realized {trailing_days}d trailing correlation?):")
            print(f"  n={len(sanity)}  Spearman rho={rho:.3f}  p={p:.6f}")
        else:
            print(f"CHECK 1: too few overlapping rows (n={len(sanity)}), skipping")

        # --- Check 2: does elevated density/mean_abs_partial_corr predict
        # elevated FORWARD realized vol / a bigger forward drawdown? ---
        merged = pd.DataFrame({
            "density": density["density"],
            "mean_abs_partial_corr": density["mean_abs_partial_corr"],
            "fwd_vol": fwd_vol,
            "fwd_worst_day": fwd_maxdd,
        }).dropna()

        print("\nCHECK 2 (does it predict forward realized risk?):")
        report("FULL HISTORY", merged)
        report("RECENT (2023+)", merged[merged.index >= RECENT_CUTOFF])

    print("\nALL_DONE")


if __name__ == "__main__":
    main()
