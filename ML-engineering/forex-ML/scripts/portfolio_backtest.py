"""Portfolio-level backtest combining multiple pairs' independent, validated
models -- every prior check in this project evaluated one pair in isolation;
this answers whether trading several of them together actually helps (real
diversification -- low/negative correlation between pairs' daily P&L) or just
compounds the same risk (correlation near +1).

Usage (run from forex-strategy's directory/venv -- imports forex_strategy.backtest
and trade_simulator.backtest/trade_simulator.portfolio):
    cd ../forex-strategy
    uv run python ../forex-ML/scripts/portfolio_backtest.py \\
        --params USD/CHF=<path> USD/JPY=<path> --weights USD/CHF=0.5 USD/JPY=0.5 \\
        --min-confidence 0.50

Each pair is evaluated using ITS OWN currently-adopted params file (so USD/CHF's
own no-daily-trend/with-Hurst config and USD/JPY's own with-daily-trend/with-Hurst
config can be passed as-is) across the same 5-fold sliding-window setup used
throughout this project, at one shared confidence threshold. Per-trade results
are bucketed into daily P&L (America/New_York, matching this project's rollover
convention) before combining, since a trade's P&L is realized at its exit, not
spread continuously across its holding period.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from forex_ml.config import load_params
from forex_ml.data.splitting import TimeSeriesSplitter, load_and_stack
from forex_ml.data.swap_rates import resolve_swap_cost_pct_per_night
from forex_ml.paths import non_time_series_parquet_path, pair_key, time_series_parquet_path
from forex_ml.spark_session import build_spark_session

from trade_simulator.backtest import simulate_trades
from trade_simulator.portfolio import (
    bucket_trades_by_day,
    combine_portfolio_daily_pnl,
    max_drawdown,
    pairwise_correlation,
    sharpe_ratio,
)

from forex_strategy.backtest import predicted_classes_to_positions

GRANULARITY = "H1"
SEED = 0


def _parse_kv_pairs(pairs: list[str]) -> dict[str, str]:
    result = {}
    for item in pairs:
        key, _, value = item.partition("=")
        result[key] = value
    return result


def _pair_daily_pnl(instrument: str, params_path: str, min_confidence: float, spark_memory: str) -> pd.Series:
    from forex.eda.eda_config.eda_config import granularity_to_seconds_map

    params = load_params(params_path)
    key = pair_key(instrument, GRANULARITY, params.feature.n_back, params.feature.lookahead)
    spark = build_spark_session(f"forex-ml-portfolio-{instrument.replace('/', '_')}", memory=spark_memory)

    pdf, pdf_non_time_series = load_and_stack(
        spark,
        str(time_series_parquet_path(params.feature.output_dir, key)),
        str(non_time_series_parquet_path(params.feature.output_dir, key)),
        params.split.columns_x,
    )
    resolved_long_swap, resolved_short_swap = resolve_swap_cost_pct_per_night(
        instrument, params.split.swap_cost_pct_per_night,
    )
    splitter = TimeSeriesSplitter(
        pdf, pdf_non_time_series, instrument, GRANULARITY,
        columns_x_components=params.split.columns_x,
        profit_take_pct=params.split.profit_take_pct,
        stop_loss_pct=params.split.stop_loss_pct,
        max_holding_bars=params.split.max_holding_bars,
        long_swap_cost_pct_per_night=resolved_long_swap,
        short_swap_cost_pct_per_night=resolved_short_swap,
    )
    purge_bars = max(params.feature.n_back, params.split.max_holding_bars)
    folds = splitter.rolling_folds(
        n_folds=5, min_train_bars=10000, val_bars=2000, test_bars=2000,
        window="sliding", purge_bars=purge_bars,
    )
    granularity_seconds = float(granularity_to_seconds_map[GRANULARITY])

    all_entry_timestamps = []
    all_net_pnl_pct = []
    fold_accs = []

    for fold_splits in folds:
        n_train, n_back, n_features = fold_splits.train["M"].shape
        X_train = fold_splits.train["M"].reshape(n_train, n_back * n_features)
        X_val = fold_splits.val["M"].reshape(fold_splits.val["M"].shape[0], n_back * n_features)
        X_test = fold_splits.test["M"].reshape(fold_splits.test["M"].shape[0], n_back * n_features)
        y_train = np.argmax(fold_splits.train["y"], axis=1)
        y_val = np.argmax(fold_splits.val["y"], axis=1)
        y_test = np.argmax(fold_splits.test["y"], axis=1)

        clf = HistGradientBoostingClassifier(random_state=SEED, early_stopping=True, validation_fraction=0.15)
        X_fit = np.concatenate([X_train, X_val], axis=0)
        y_fit = np.concatenate([y_train, y_val], axis=0)
        clf.fit(X_fit, y_fit)
        fold_accs.append((np.argmax(clf.predict_proba(X_test), axis=1) == y_test).mean())

        pred_proba = clf.predict_proba(X_test)
        entry_timestamp = fold_splits.test["timestamp"]
        long_exit_timestamp = entry_timestamp + fold_splits.test["long_exit_bar_offset"] * granularity_seconds
        short_exit_timestamp = entry_timestamp + fold_splits.test["short_exit_bar_offset"] * granularity_seconds

        positions = predicted_classes_to_positions(pred_proba, min_confidence=min_confidence)
        result = simulate_trades(
            positions, fold_splits.test["long_raw_return_pct"], fold_splits.test["short_raw_return_pct"],
            fold_splits.test["spread"], fold_splits.test["price"],
            entry_timestamp=entry_timestamp,
            long_exit_timestamp=long_exit_timestamp, short_exit_timestamp=short_exit_timestamp,
            long_swap_cost_pct_per_night=fold_splits.long_swap_cost_pct_per_night,
            short_swap_cost_pct_per_night=fold_splits.short_swap_cost_pct_per_night,
        )
        # simulate_trades doesn't expose its internal is_trade mask -- with the
        # default position_size (all ones, no flatten_before_rollover) used here,
        # sized_positions == positions, so this reproduces that mask exactly to
        # pull out each closed trade's own entry timestamp, aligned with
        # per_trade_net_pnl_pct's order (both derived from the same boolean mask
        # over the same underlying arrays).
        is_trade = positions != 0
        all_entry_timestamps.append(entry_timestamp[is_trade])
        all_net_pnl_pct.append(result.per_trade_net_pnl_pct)

    print(f"{instrument}: mean fold accuracy = {np.mean(fold_accs):.4f}, "
          f"total trades = {sum(len(a) for a in all_net_pnl_pct)}", flush=True)

    combined_timestamps = np.concatenate(all_entry_timestamps)
    combined_pnl = np.concatenate(all_net_pnl_pct)
    return bucket_trades_by_day(combined_timestamps, combined_pnl)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", nargs="+", required=True, help="INSTRUMENT=path/to/params.yaml, one per pair")
    parser.add_argument("--weights", nargs="+", required=True, help="INSTRUMENT=weight (should sum to 1.0), one per pair")
    parser.add_argument("--min-confidence", type=float, default=0.50)
    parser.add_argument("--spark-memory", default="24g")
    args = parser.parse_args()

    params_by_instrument = _parse_kv_pairs(args.params)
    weights = {k: float(v) for k, v in _parse_kv_pairs(args.weights).items()}

    daily_pnl_by_pair = {
        instrument: _pair_daily_pnl(instrument, params_path, args.min_confidence, args.spark_memory)
        for instrument, params_path in params_by_instrument.items()
    }

    print(f"\n{'=' * 70}\nPER-PAIR STANDALONE STATS (min_confidence={args.min_confidence})\n{'=' * 70}", flush=True)
    for instrument, daily_pnl in daily_pnl_by_pair.items():
        print(f"  {instrument:10s}  n_trading_days={len(daily_pnl):4d}  "
              f"total_net_pnl_pct={daily_pnl.sum():8.3f}  "
              f"sharpe={sharpe_ratio(daily_pnl):6.3f}  "
              f"max_drawdown={max_drawdown(daily_pnl):7.3f}", flush=True)

    combined = combine_portfolio_daily_pnl(daily_pnl_by_pair, weights)
    print(f"\n{'=' * 70}\nCOMBINED PORTFOLIO (weights={weights})\n{'=' * 70}", flush=True)
    print(f"  n_trading_days={len(combined)}  total_net_pnl_pct={combined.sum():.3f}  "
          f"sharpe={sharpe_ratio(combined):.3f}  max_drawdown={max_drawdown(combined):.3f}", flush=True)

    equal_weighted_avg_sharpe = np.mean([sharpe_ratio(s) for s in daily_pnl_by_pair.values()])
    print(f"\n  (unweighted average of each pair's own standalone Sharpe: {equal_weighted_avg_sharpe:.3f} --"
          f" combined > this means real diversification benefit, not just riding the same risk twice)", flush=True)

    print(f"\n{'=' * 70}\nPAIRWISE CORRELATION (daily P&L)\n{'=' * 70}", flush=True)
    print(pairwise_correlation(daily_pnl_by_pair), flush=True)

    print("\nALL_DONE", flush=True)


if __name__ == "__main__":
    main()
