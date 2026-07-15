"""Grouped permutation importance + plots for one (instrument, granularity, params)
combination, using forex_ml.evaluation.feature_importance
(compute_permutation_importance, plot_feature_importances).

Usage:
    uv run python scripts/permutation_importance_report.py \\
        --params params.yaml --instrument USD/CHF --granularity H1 --output-dir .

Two groupings of the flattened (n_back x n_features) input, since permuting each
individual column separately would be slow and mostly uninformative (many
near-duplicate lags per feature):

1. By base feature name: shuffle all `n_back` lags of one named feature (e.g. all
   of "rsi_12"'s columns) together across test rows, holding every other feature
   fixed -- which of the engineered features matters most.
2. By recency bucket (5 equal-width groups spanning the lookback window): shuffle
   all features' columns within one lag range together -- do recent bars matter
   more than distant ones. Bucket 0 = oldest, bucket 4 = most recent (closest to
   the entry/prediction point) -- see forex_ml/data/splitting.py's
   _stack_time_series_fn for the lag-ordering proof (M[t][f], t ascending from
   oldest to newest).

Trains across 5 sliding-window multi-window folds (forex_ml.data.splitting's
rolling_folds) and averages results across them, rather than trusting a single
train/test split.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

from forex_ml.config import load_params
from forex_ml.data.splitting import TimeSeriesSplitter, load_and_stack
from forex_ml.data.swap_rates import resolve_swap_cost_pct_per_night
from forex_ml.evaluation.feature_importance import compute_permutation_importance, plot_feature_importances
from forex_ml.paths import non_time_series_parquet_path, pair_key, time_series_parquet_path
from forex_ml.spark_session import build_spark_session

N_REPEATS = 5
N_RECENCY_BUCKETS = 5
RNG_SEED = 0


def _pool_across_folds(fold_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Each fold contributes one importance_mean per feature/group; pool those
    per-fold means into a final mean/std -- fold-to-fold spread, not the
    within-fold n_repeats spread `compute_permutation_importance` already reports
    per fold."""
    combined = pd.concat(fold_dfs)
    pooled = combined.groupby("feature")["importance_mean"].agg(["mean", "std"]).reset_index()
    return pooled.rename(columns={"mean": "importance_mean", "std": "importance_std"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", required=True, help="Path to a forex_ml params.yaml")
    parser.add_argument("--instrument", required=True, help="e.g. USD/CHF")
    parser.add_argument("--granularity", required=True, help="e.g. H1")
    parser.add_argument("--output-dir", default=".", help="Where to write the two PNG plots (default: cwd)")
    parser.add_argument("--spark-memory", default="24g")
    args = parser.parse_args()

    label = args.instrument.replace("/", "_")
    spark = build_spark_session(f"forex-ml-permutation-importance-{label}", memory=args.spark_memory)

    params = load_params(args.params)
    columns_x = params.split.columns_x
    n_features = len(columns_x)
    key = pair_key(args.instrument, args.granularity, params.feature.n_back, params.feature.lookahead)

    pdf, pdf_non_time_series = load_and_stack(
        spark,
        str(time_series_parquet_path(params.feature.output_dir, key)),
        str(non_time_series_parquet_path(params.feature.output_dir, key)),
        columns_x,
    )
    resolved_long_swap, resolved_short_swap = resolve_swap_cost_pct_per_night(
        args.instrument, params.split.swap_cost_pct_per_night,
    )
    splitter = TimeSeriesSplitter(
        pdf, pdf_non_time_series, args.instrument, args.granularity,
        columns_x_components=columns_x,
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

    by_feature_folds = []
    by_bucket_folds = []
    fold_accs = []

    for fold_idx, fold_splits in enumerate(folds):
        n_train, n_back, _ = fold_splits.train["M"].shape
        X_train = fold_splits.train["M"].reshape(n_train, n_back * n_features)
        X_val = fold_splits.val["M"].reshape(fold_splits.val["M"].shape[0], n_back * n_features)
        X_test = fold_splits.test["M"].reshape(fold_splits.test["M"].shape[0], n_back * n_features)
        y_train = np.argmax(fold_splits.train["y"], axis=1)
        y_val = np.argmax(fold_splits.val["y"], axis=1)
        y_test = np.argmax(fold_splits.test["y"], axis=1)

        clf = HistGradientBoostingClassifier(random_state=0, early_stopping=True, validation_fraction=0.15)
        X_fit = np.concatenate([X_train, X_val], axis=0)
        y_fit = np.concatenate([y_train, y_val], axis=0)
        clf.fit(X_fit, y_fit)

        baseline_acc = accuracy_score(y_test, clf.predict(X_test))
        fold_accs.append(baseline_acc)
        print(f"Fold {fold_idx}: baseline_test_accuracy={baseline_acc:.4f}", flush=True)

        feature_groups = {
            name: [t * n_features + f_idx for t in range(n_back)] for f_idx, name in enumerate(columns_x)
        }
        by_feature_folds.append(compute_permutation_importance(
            clf, X_test, y_test, feature_names=columns_x, feature_groups=feature_groups,
            n_repeats=N_REPEATS, random_state=RNG_SEED,
        ))

        bucket_size = n_back // N_RECENCY_BUCKETS
        bucket_groups = {
            f"bucket_{b}": [
                t * n_features + f
                for t in range(b * bucket_size, n_back if b == N_RECENCY_BUCKETS - 1 else (b + 1) * bucket_size)
                for f in range(n_features)
            ]
            for b in range(N_RECENCY_BUCKETS)
        }
        by_bucket_folds.append(compute_permutation_importance(
            clf, X_test, y_test, feature_names=list(bucket_groups), feature_groups=bucket_groups,
            n_repeats=N_REPEATS, random_state=RNG_SEED,
        ))

    print(f"\nmean fold baseline accuracy = {np.mean(fold_accs):.4f}", flush=True)

    by_feature = _pool_across_folds(by_feature_folds)
    by_bucket = _pool_across_folds(by_bucket_folds)

    print("\n--- feature importance (mean accuracy drop across folds, sorted) ---", flush=True)
    for _, row in by_feature.sort_values("importance_mean", ascending=False).iterrows():
        print(f"  {row['feature']:28s}  mean_drop={row['importance_mean']:+.5f}  std={row['importance_std']:.5f}", flush=True)

    print(f"\n--- recency-bucket importance (bucket_0=oldest .. bucket_{N_RECENCY_BUCKETS - 1}=most recent) ---", flush=True)
    for _, row in by_bucket.sort_values("feature").iterrows():
        print(f"  {row['feature']:12s}  mean_drop={row['importance_mean']:+.5f}  std={row['importance_std']:.5f}", flush=True)

    feature_plot_path = f"{args.output_dir}/{label}_feature_importance.png"
    bucket_plot_path = f"{args.output_dir}/{label}_recency_importance.png"
    plot_feature_importances(by_feature, title=f"{args.instrument}: feature importance", output_path=feature_plot_path)
    plot_feature_importances(by_bucket, title=f"{args.instrument}: recency-bucket importance", output_path=bucket_plot_path)
    print(f"\nSaved plots to {feature_plot_path} and {bucket_plot_path}", flush=True)


if __name__ == "__main__":
    main()
