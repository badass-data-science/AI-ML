"""Rolling (walk-forward) cross-validation — a robustness DIAGNOSTIC, not a model
selection or deployment strategy.

Rather than trusting a single train/val/test split — which could be idiosyncratic to
whichever slice of history it happened to land on — this walks a fixed-size train/
val/test window forward through the timeline, retraining fresh each time, and reports
the DISTRIBUTION of test-set results across folds rather than a single number.

Two window types are supported (see TimeSeriesSplitter.rolling_folds), both walking
forward by one test-block each fold:
  - "sliding": the training block has a fixed length and slides forward with the
    fold, so every fold trains on a comparably-sized, comparably-recent history.
    More robust to regime change — older data ages out — at the cost of using less
    data per fold than is actually available by the final fold.
  - "expanding": the training block always starts at the very first bar and grows by
    one test-block's worth each fold, so later folds see strictly more history. Uses
    all available data, at the cost of assuming that older data is still as relevant
    as recent data — a stronger stationarity assumption.

Each fold is trained and logged to its OWN MLflow experiment (`<experiment>-rolling-
cv` by default), tagged with its fold index and window type, and NOT registered in
the model registry — these are diagnostic runs, not deployment candidates. Keeping
them in a separate experiment means forex_ml.evaluation.multiple_comparisons's BH-FDR
pool (which scans one experiment for one "official" run per (pair, configuration))
never sees them and is never distorted by them.

This is purely additive: it doesn't change what gets deployed, registered, or fed
into the multiple-comparisons report. Two related but heavier extensions are
possible future next steps, not built here — see README.md's Diagnostics section:
  - using fold results to choose between competing architectures/hyperparameters
    (rather than just reporting how stable ONE configuration is), or
  - an actual walk-forward RETRAINING strategy, where each fold's model is a real
    deployment candidate for its period rather than a diagnostic artifact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pyspark.sql import SparkSession

from forex_ml.config import load_params
from forex_ml.data.splitting import TimeSeriesSplitter, load_and_stack
from forex_ml.paths import non_time_series_parquet_path, pair_key, time_series_parquet_path
from forex_ml.spark_session import DEFAULT_SPARK_MEMORY, build_spark_session
from forex_ml.training.train import train_and_evaluate


def _aggregate(values: list[float]) -> dict:
    arr = np.array(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def run_rolling_cv(
    spark: SparkSession,
    instrument: str,
    granularity: str,
    n_folds: int,
    min_train_bars: int,
    val_bars: int,
    test_bars: int,
    window: str = "sliding",
    purge_bars: int | None = None,
    params_path: str | None = None,
) -> dict:
    """Runs `n_folds` walk-forward folds for one (instrument, granularity) pair
    against its real Stage-1 output, and reports the distribution of test-set
    results across them — for the LSTM and both baselines, so a wide LSTM spread
    that still clears the baseline every fold reads differently than one that
    doesn't. See the module docstring for "sliding" vs "expanding".

    `purge_bars` defaults to `max(n_back, lookahead)` — the same choice
    `split_flow.py` makes for the single official split, for the same reason (a
    window/label can reach that far across a boundary).
    """
    params = load_params(params_path) if params_path else load_params()
    key = pair_key(instrument, granularity, params.feature.n_back, params.feature.lookahead)

    pdf, pdf_non_time_series = load_and_stack(
        spark,
        str(time_series_parquet_path(params.feature.output_dir, key)),
        str(non_time_series_parquet_path(params.feature.output_dir, key)),
        params.split.columns_x,
        params.split.column_y,
    )
    splitter = TimeSeriesSplitter(
        pdf, pdf_non_time_series, instrument, granularity,
        columns_x_components=params.split.columns_x,
        class_cutoff_percentiles=params.split.class_cutoff_percentiles,
        column_y=params.split.column_y,
    )

    resolved_purge_bars = (
        purge_bars if purge_bars is not None else max(params.feature.n_back, params.feature.lookahead)
    )
    folds = splitter.rolling_folds(
        n_folds=n_folds, min_train_bars=min_train_bars, val_bars=val_bars, test_bars=test_bars,
        window=window, purge_bars=resolved_purge_bars,
    )

    experiment_name = f"{params.train.mlflow_experiment_name}-rolling-cv"
    metric_key = params.train.metrics[0]  # whatever model.evaluate reports the LSTM's own score as

    fold_results = []
    for i, fold_splits in enumerate(folds):
        result = train_and_evaluate(
            fold_splits, params.train, instrument, granularity, Path(params.feature.output_dir),
            params.feature.n_back, params.feature.lookahead, params.split.column_y,
            experiment_name=experiment_name,
            register_model=False,
            extra_params={"fold_index": i, "window_type": window, "diagnostic": "rolling_cv"},
            run_name_suffix=f"fold{i}",
        )
        fold_results.append(result)

    lstm_scores = [r[metric_key] for r in fold_results]
    majority_scores = [r["baseline_majority_accuracy"] for r in fold_results]
    persistence_scores = [r["baseline_persistence_accuracy"] for r in fold_results]

    return {
        "instrument": instrument,
        "granularity": granularity,
        "window": window,
        "n_folds": n_folds,
        "metric": metric_key,
        "fold_lstm_scores": lstm_scores,
        "fold_baseline_majority_scores": majority_scores,
        "fold_baseline_persistence_scores": persistence_scores,
        "lstm": _aggregate(lstm_scores),
        "baseline_majority": _aggregate(majority_scores),
        "baseline_persistence": _aggregate(persistence_scores),
    }


def _print_report(report: dict) -> None:
    print(
        f"{report['instrument']} {report['granularity']} — {report['n_folds']} "
        f"'{report['window']}'-window rolling folds (metric: {report['metric']})\n"
    )
    print(f"  per-fold LSTM {report['metric']}: {[round(v, 3) for v in report['fold_lstm_scores']]}\n")
    for label, key in [
        ("LSTM", "lstm"),
        ("majority baseline", "baseline_majority"),
        ("persistence baseline", "baseline_persistence"),
    ]:
        stats = report[key]
        print(
            f"  {label:22s} mean={stats['mean']:.3f}  std={stats['std']:.3f}  "
            f"min={stats['min']:.3f}  max={stats['max']:.3f}"
        )
    if report["lstm"]["mean"] <= report["baseline_majority"]["mean"]:
        print(
            "\n  NOTE: mean LSTM performance across folds does not exceed the majority "
            "baseline's -- this configuration isn't robustly beating a trivial rule across "
            "time periods, not just in one split."
        )
    if report["lstm"]["std"] > (report["lstm"]["mean"] - report["baseline_majority"]["mean"]):
        print(
            "\n  NOTE: the LSTM's fold-to-fold variability is larger than its average margin "
            "over the majority baseline -- a single train/val/test split could easily have "
            "landed on an unusually good OR unusually bad fold for this configuration."
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Walk-forward rolling cross-validation: a robustness diagnostic across time "
                     "periods for one (instrument, granularity) pair. Does not affect the model "
                     "registry or the multiple-comparisons report."
    )
    parser.add_argument("--instrument", required=True, help="e.g. EUR/USD")
    parser.add_argument("--granularity", required=True, help="e.g. H1")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--window", choices=["sliding", "expanding"], default="sliding")
    parser.add_argument(
        "--min-train-bars", type=int, required=True,
        help="Training block size in bars: fixed for 'sliding'; the FIRST fold's size for "
             "'expanding' (it then grows by --test-bars each subsequent fold)",
    )
    parser.add_argument("--val-bars", type=int, required=True)
    parser.add_argument("--test-bars", type=int, required=True)
    parser.add_argument(
        "--purge-bars", type=int, default=None,
        help="Default: max(feature.n_back, feature.lookahead) from params.yaml",
    )
    parser.add_argument("--params", default=None, help="Path to params.yaml (default: repo root)")
    parser.add_argument(
        "--spark-memory", default=DEFAULT_SPARK_MEMORY,
        help=f"spark.driver.memory / spark.executor.memory / spark.driver.maxResultSize (default: {DEFAULT_SPARK_MEMORY})",
    )
    args = parser.parse_args()

    spark = build_spark_session("forex-ml-rolling-cv", memory=args.spark_memory)
    report = run_rolling_cv(
        spark, args.instrument, args.granularity, args.n_folds,
        args.min_train_bars, args.val_bars, args.test_bars,
        window=args.window, purge_bars=args.purge_bars, params_path=args.params,
    )
    _print_report(report)


if __name__ == "__main__":
    main()
