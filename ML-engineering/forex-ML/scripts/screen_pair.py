"""Formal pair-screening protocol, runnable against any (instrument, granularity,
params) combination -- see forex_ml.evaluation.pair_screening for the rationale.

Usage (run from forex-strategy's own directory/venv -- needs forex_strategy.backtest
and trade_simulator.backtest, see below):
    cd ../forex-strategy
    uv run python ../forex-ML/scripts/screen_pair.py \\
        --params params.yaml --instrument AUD/NZD --granularity H1 \\
        [--auto-calibrate] [--seeds 0,7,17,54,100,2026] [--alpha 0.05] \\
        [--min-timestamp 2023-01-01]

Protocol, in order:
1. (Optional, --auto-calibrate) Calibrate profit_take_pct/stop_loss_pct from the
   pair's own real median 24-bar move, rather than reusing another pair's
   threshold or a hand-picked guess.
2. Multi-window backtest: 5 sliding-window folds (10000/2000/2000 bars, purged),
   HistGradientBoostingClassifier(random_state=0), pooled across all 5 folds' test
   rows, at 5 confidence thresholds. Folds are always anchored at the EARLIEST
   timestamp present in the data and step forward a small, fixed number of bars
   each -- with these fold sizes that's only ~2.5 years of coverage, so on an
   unfiltered multi-year Stage-1 file the folds sit wherever the data happens to
   start (e.g. 2016-2017 for a pair whose Stage 1 starts at
   min_training_timestamp=2015-01-01), NOT anywhere near "now". Pass
   --min-timestamp to validate against a recent window instead.
3. Verdict: does any threshold clear Benjamini-Hochberg-corrected significance
   (win-rate OR per-trade payoff-asymmetry path) with positive pooled net P&L?
   Deliberately does NOT gate on single-window accuracy or a regime-shift check
   first -- this project found both AUD/USD (looked credible, failed) and
   USD/JPY (looked weak, passed) contradicted that kind of pre-screening.
4. If the pair passes: automatically rerun the SAME folds under multiple
   classifier seeds, and report how many are BH-significant at each threshold
   that passed step 3 -- a pair whose result depends on getting lucky with one
   seed is a materially weaker candidate than one that holds up broadly.

Requires forex_strategy.backtest.predicted_classes_to_positions (forex-ML-
specific: translates a 3-class prediction into a position) and
trade_simulator.backtest.simulate_trades (model-agnostic: positions -> P&L,
see github.com/badass-data-science/forex-trade-simulation-inator) -- run this
from forex-strategy's directory so all three packages are importable, the
same convention used by every other multi-window script this project has run.
"""

from __future__ import annotations

import argparse
import datetime

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator

from forex_ml.config import load_params
from forex_ml.data.splitting import TimeSeriesSplitter, load_and_stack
from forex_ml.data.swap_rates import resolve_swap_cost_pct_per_night
from forex_ml.evaluation.pair_screening import (
    calibrate_symmetric_barrier_pct,
    evaluate_screening_verdict,
    pool_fold_results,
)
from forex_ml.paths import non_time_series_parquet_path, pair_key, time_series_parquet_path
from forex_ml.spark_session import build_spark_session

from trade_simulator.backtest import simulate_trades

from forex_strategy.backtest import predicted_classes_to_positions

DEFAULT_SEEDS = [0, 7, 17, 54, 100, 2026]
CONFIDENCE_THRESHOLDS = [0.0, 0.40, 0.45, 0.50, 0.55]


def _parse_min_timestamp(raw: str) -> float:
    """Accepts either a unix-epoch-seconds float or an ISO date/datetime string,
    matching how `feature.min_training_timestamp` is already interpreted elsewhere
    in this project (naive local time, not UTC)."""
    try:
        return float(raw)
    except ValueError:
        return datetime.datetime.fromisoformat(raw).timestamp()


def _build_folds(spark, params, instrument: str, granularity: str, min_timestamp: float | None = None):
    """`min_timestamp` (unix epoch seconds), if given, drops every row strictly
    before it BEFORE folds are built. This matters because `rolling_folds`'s
    "sliding" window always anchors fold 0 at the EARLIEST timestamp remaining in
    the frame and steps forward a small, fixed number of bars per fold -- with
    n_folds=5 and test_bars=2000 (~83 H1 days/fold), the 5 folds only ever cover
    ~2.5 years from wherever they start. Left unfiltered, that's always the first
    ~2.5 years of whatever's in Stage 1 (e.g. 2016-2017 for a pair whose Stage 1
    starts at min_training_timestamp=2015-01-01) -- NEVER anywhere near "now",
    no matter how much more recent data Stage 1 actually contains. Pass a recent
    cutoff to validate against current market conditions instead."""
    key = pair_key(instrument, granularity, params.feature.n_back, params.feature.lookahead)
    pdf, pdf_non_time_series = load_and_stack(
        spark,
        str(time_series_parquet_path(params.feature.output_dir, key)),
        str(non_time_series_parquet_path(params.feature.output_dir, key)),
        params.split.columns_x,
    )
    if min_timestamp is not None:
        pdf = pdf[pdf["unix_epoch_s"] >= min_timestamp].reset_index(drop=True)
        pdf_non_time_series = pdf_non_time_series[pdf_non_time_series["unix_epoch_s"] >= min_timestamp].reset_index(drop=True)
    resolved_long_swap, resolved_short_swap = resolve_swap_cost_pct_per_night(
        instrument, params.split.swap_cost_pct_per_night,
    )
    profit_take_pct, stop_loss_pct = params.split.profit_take_pct, params.split.stop_loss_pct
    splitter = TimeSeriesSplitter(
        pdf, pdf_non_time_series, instrument, granularity,
        columns_x_components=params.split.columns_x,
        profit_take_pct=profit_take_pct, stop_loss_pct=stop_loss_pct,
        max_holding_bars=params.split.max_holding_bars,
        long_swap_cost_pct_per_night=resolved_long_swap,
        short_swap_cost_pct_per_night=resolved_short_swap,
    )
    purge_bars = max(params.feature.n_back, params.split.max_holding_bars)
    folds = splitter.rolling_folds(
        n_folds=5, min_train_bars=10000, val_bars=2000, test_bars=2000,
        window="sliding", purge_bars=purge_bars,
    )
    return folds, splitter


def _fold_simulation_results(
    folds, granularity_seconds: float, seed: int, calibrate: str | None = None,
) -> dict[float, list]:
    results_by_threshold: dict[float, list] = {thr: [] for thr in CONFIDENCE_THRESHOLDS}
    for fold_splits in folds:
        n_train, n_back, n_features = fold_splits.train["M"].shape
        X_train = fold_splits.train["M"].reshape(n_train, n_back * n_features)
        X_val = fold_splits.val["M"].reshape(fold_splits.val["M"].shape[0], n_back * n_features)
        X_test = fold_splits.test["M"].reshape(fold_splits.test["M"].shape[0], n_back * n_features)
        y_train = np.argmax(fold_splits.train["y"], axis=1)
        y_val = np.argmax(fold_splits.val["y"], axis=1)

        clf = HistGradientBoostingClassifier(random_state=seed, early_stopping=True, validation_fraction=0.15)

        if calibrate is None:
            X_fit = np.concatenate([X_train, X_val], axis=0)
            y_fit = np.concatenate([y_train, y_val], axis=0)
            clf.fit(X_fit, y_fit)
            pred_proba = clf.predict_proba(X_test)
        else:
            # val must stay genuinely unseen by the base fit -- calibrating
            # against predictions the model already memorized would understate
            # how overconfident it really is on data it hasn't seen.
            clf.fit(X_train, y_train)
            calibrated_clf = CalibratedClassifierCV(FrozenEstimator(clf), method=calibrate)
            calibrated_clf.fit(X_val, y_val)
            pred_proba = calibrated_clf.predict_proba(X_test)
        entry_timestamp = fold_splits.test["timestamp"]
        long_exit_timestamp = entry_timestamp + fold_splits.test["long_exit_bar_offset"] * granularity_seconds
        short_exit_timestamp = entry_timestamp + fold_splits.test["short_exit_bar_offset"] * granularity_seconds

        for min_confidence in CONFIDENCE_THRESHOLDS:
            positions = predicted_classes_to_positions(pred_proba, min_confidence=min_confidence)
            result = simulate_trades(
                positions, fold_splits.test["long_raw_return_pct"], fold_splits.test["short_raw_return_pct"],
                fold_splits.test["spread"], fold_splits.test["price"],
                entry_timestamp=entry_timestamp,
                long_exit_timestamp=long_exit_timestamp, short_exit_timestamp=short_exit_timestamp,
                long_swap_cost_pct_per_night=fold_splits.long_swap_cost_pct_per_night,
                short_swap_cost_pct_per_night=fold_splits.short_swap_cost_pct_per_night,
            )
            results_by_threshold[min_confidence].append(result)
    return results_by_threshold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", required=True, help="Path to a forex_ml params.yaml")
    parser.add_argument("--instrument", required=True, help="e.g. AUD/NZD")
    parser.add_argument("--granularity", required=True, help="e.g. H1")
    parser.add_argument("--auto-calibrate", action="store_true",
                         help="Recompute profit_take_pct/stop_loss_pct from this pair's own real "
                              "median move over max_holding_bars, ignoring the params file's values")
    parser.add_argument("--min-timestamp", default=None,
                         help="ISO date (e.g. 2023-01-01) or unix-epoch-seconds float. Drops rows "
                              "before this BEFORE folds are built, so the 5 folds are anchored here "
                              "instead of at Stage 1's earliest bar -- use to validate against recent "
                              "market conditions rather than whatever period the data happens to start "
                              "at (see _build_folds docstring).")
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                         help="Comma-separated seeds for the stability check")
    parser.add_argument("--proba-calibration", choices=["sigmoid", "isotonic"], default=None,
                         help="Post-hoc predict_proba calibration (Platt/sigmoid or isotonic), fit on "
                              "each fold's val split, held genuinely out of the base classifier's own "
                              "training -- unrelated to --auto-calibrate, which tunes barrier pct, not "
                              "probabilities. Default: no calibration (matches every prior screening run).")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--spark-memory", default="24g")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    min_timestamp = _parse_min_timestamp(args.min_timestamp) if args.min_timestamp is not None else None

    from forex.eda.eda_config.eda_config import granularity_to_seconds_map
    granularity_seconds = float(granularity_to_seconds_map[args.granularity])

    params = load_params(args.params)

    # One Spark session for the whole run -- SparkSession.builder...getOrCreate()
    # silently REUSES an already-created session's config (memory settings
    # included) rather than honoring new ones, so building a second, differently
    # sized session later (e.g. for the main folds after an auto-calibrate probe)
    # would silently keep running under the FIRST session's memory limit. Bit us
    # directly: an 8g calibration-probe session followed by a requested 24g main
    # session actually ran the whole heavy computation at 8g and OOM'd.
    key = pair_key(args.instrument, args.granularity, params.feature.n_back, params.feature.lookahead)
    spark = build_spark_session(f"forex-ml-screen-{args.instrument.replace('/', '_')}", memory=args.spark_memory)

    if args.auto_calibrate:
        # mid_close is a COLUMNS_PASSTHROUGH field, which load_and_stack only
        # attaches to its FIRST return value (the time-series frame carrying "X"),
        # not the second (non-time-series) one -- using the wrong one here 404s
        # with a plain KeyError.
        pdf_probe, _ = load_and_stack(
            spark,
            str(time_series_parquet_path(params.feature.output_dir, key)),
            str(non_time_series_parquet_path(params.feature.output_dir, key)),
            params.split.columns_x,
        )
        pair_frame = pdf_probe[
            (pdf_probe["instrument"] == args.instrument)
            & (pdf_probe["granularity"] == args.granularity)
        ].sort_values("unix_epoch_s")
        calibrated_pct = calibrate_symmetric_barrier_pct(
            pair_frame["mid_close"].to_numpy(), holding_bars=params.split.max_holding_bars,
        )
        print(f"Auto-calibrated profit_take_pct=stop_loss_pct={calibrated_pct} "
              f"(median {params.split.max_holding_bars}-bar move)", flush=True)
        params.split.profit_take_pct = calibrated_pct
        params.split.stop_loss_pct = calibrated_pct

    min_ts_label = (
        datetime.datetime.fromtimestamp(min_timestamp).isoformat() if min_timestamp is not None else "earliest available"
    )
    print(f"\n{'=' * 70}\nSCREENING {args.instrument} {args.granularity}  "
          f"(profit_take_pct={params.split.profit_take_pct}, stop_loss_pct={params.split.stop_loss_pct}, "
          f"proba_calibration={args.proba_calibration}, folds anchored from={min_ts_label})\n{'=' * 70}", flush=True)

    folds, _ = _build_folds(spark, params, args.instrument, args.granularity, min_timestamp=min_timestamp)
    print(f"Got {len(folds)} folds", flush=True)

    seed0_results = _fold_simulation_results(folds, granularity_seconds, seed=0, calibrate=args.proba_calibration)
    pooled = {thr: pool_fold_results(thr, results) for thr, results in seed0_results.items()}
    for thr, r in sorted(pooled.items()):
        print(f"  conf={thr:.2f}  trades={r.total_trades:5d}  win_rate={r.pooled_win_rate:.3f}  "
              f"win_rate_p={r.win_rate_p_value:.4f}  net_pnl_pct={r.total_net_pnl_pct:8.3f}  "
              f"per_trade_t_p={r.per_trade_t_p_value:.4f}  per_trade_w_p={r.per_trade_wilcoxon_p_value:.4f}", flush=True)

    verdict = evaluate_screening_verdict(pooled, alpha=args.alpha)
    print(f"\nVERDICT: {'PASS' if verdict.passed else 'FAIL'} -- {verdict.reason}", flush=True)
    if verdict.passed:
        print(f"Passing thresholds: {verdict.passing_thresholds}", flush=True)

        print(f"\n{'=' * 70}\nSEED-STABILITY CHECK (seeds={seeds})\n{'=' * 70}", flush=True)
        seed_pass_counts = {thr: 0 for thr in verdict.passing_thresholds}
        for seed in seeds:
            seed_results = _fold_simulation_results(folds, granularity_seconds, seed=seed, calibrate=args.proba_calibration)
            seed_pooled = {thr: pool_fold_results(thr, results) for thr, results in seed_results.items()}
            seed_verdict = evaluate_screening_verdict(
                {thr: seed_pooled[thr] for thr in verdict.passing_thresholds}, alpha=args.alpha,
            )
            for thr in verdict.passing_thresholds:
                r = seed_pooled[thr]
                passed_here = thr in seed_verdict.passing_thresholds
                seed_pass_counts[thr] += int(passed_here)
                print(f"  seed={seed:5d}  conf={thr:.2f}  win_rate={r.pooled_win_rate:.3f}  "
                      f"net_pnl_pct={r.total_net_pnl_pct:8.3f}  passes_here={passed_here}", flush=True)

        print(f"\nSEED-STABILITY SUMMARY (of {len(seeds)} seeds):", flush=True)
        for thr, count in sorted(seed_pass_counts.items()):
            print(f"  conf={thr:.2f}: {count}/{len(seeds)} seeds pass", flush=True)

    print("\nALL_DONE", flush=True)


if __name__ == "__main__":
    main()
