"""ROC + calibration plots for one (instrument, granularity, params) combination,
using forex_ml.evaluation.classifier_diagnostics (plot_multiclass_roc,
plot_calibration_curve).

Usage:
    uv run python scripts/roc_calibration_report.py \\
        --params params.yaml --instrument USD/CHF --granularity H1 --output-dir .

Predictions are pooled across 5 sliding-window multi-window folds' test sets
before plotting (one combined ROC/calibration read across ~5x the data of a
single train/test split), rather than trusting one window's read -- also gives
the calibration curve more test rows per probability bin.
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator

from forex_ml.config import load_params
from forex_ml.data.splitting import TimeSeriesSplitter, load_and_stack
from forex_ml.data.swap_rates import resolve_swap_cost_pct_per_night
from forex_ml.evaluation.classifier_diagnostics import plot_calibration_curve, plot_multiclass_roc
from forex_ml.paths import non_time_series_parquet_path, pair_key, time_series_parquet_path
from forex_ml.spark_session import build_spark_session

CLASS_NAMES = ["short", "flat", "long"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", required=True, help="Path to a forex_ml params.yaml")
    parser.add_argument("--instrument", required=True, help="e.g. USD/CHF")
    parser.add_argument("--granularity", required=True, help="e.g. H1")
    parser.add_argument("--output-dir", default=".", help="Where to write the two PNG plots (default: cwd)")
    parser.add_argument("--proba-calibration", choices=["sigmoid", "isotonic"], default=None,
                         help="Post-hoc predict_proba calibration, fit on each fold's val split held "
                              "genuinely out of the base classifier's own training. Default: none, "
                              "raw predict_proba (matches every prior report run).")
    parser.add_argument("--spark-memory", default="24g")
    args = parser.parse_args()

    label = args.instrument.replace("/", "_")
    if args.proba_calibration:
        label = f"{label}_{args.proba_calibration}"
    spark = build_spark_session(f"forex-ml-roc-calibration-{label}", memory=args.spark_memory)

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

    pooled_y_test = []
    pooled_y_proba = []
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

        if args.proba_calibration is None:
            X_fit = np.concatenate([X_train, X_val], axis=0)
            y_fit = np.concatenate([y_train, y_val], axis=0)
            clf.fit(X_fit, y_fit)
            y_proba = clf.predict_proba(X_test)
        else:
            clf.fit(X_train, y_train)
            calibrated_clf = CalibratedClassifierCV(FrozenEstimator(clf), method=args.proba_calibration)
            calibrated_clf.fit(X_val, y_val)
            y_proba = calibrated_clf.predict_proba(X_test)
        acc = (np.argmax(y_proba, axis=1) == y_test).mean()
        fold_accs.append(acc)
        print(f"Fold {fold_idx}: test_accuracy={acc:.4f}", flush=True)

        pooled_y_test.append(y_test)
        pooled_y_proba.append(y_proba)

    y_test_all = np.concatenate(pooled_y_test)
    y_proba_all = np.concatenate(pooled_y_proba)
    print(f"\nmean fold accuracy = {np.mean(fold_accs):.4f}  (pooled n={len(y_test_all)})", flush=True)

    calibration_label = args.proba_calibration or "none"
    roc_path = f"{args.output_dir}/{label}_roc.png"
    calibration_path = f"{args.output_dir}/{label}_calibration.png"
    plot_multiclass_roc(y_test_all, y_proba_all, CLASS_NAMES, title=f"{args.instrument}: ROC (pooled across folds, proba_calibration={calibration_label})", output_path=roc_path)
    plot_calibration_curve(y_test_all, y_proba_all, CLASS_NAMES, n_bins=10, title=f"{args.instrument}: calibration (pooled across folds, proba_calibration={calibration_label})", output_path=calibration_path)
    print(f"\nSaved plots to {roc_path} and {calibration_path}", flush=True)


if __name__ == "__main__":
    main()
