"""Stage 3: train the LSTM regressor and evaluate it, with MLflow tracking + registry.

Fixes two bugs from the original lstm.py:
  - Stage 2 computes a real time-ordered train/val/test split, but the original
    script only ever loaded `train` and then carved a *second*, different
    validation set out of it with `validation_split=0.2` — `val` and `test` were
    pickled but never read. This version trains with `validation_data=` set to
    the real Stage-2 `val` split.
  - there was no held-out test evaluation at all. This version evaluates on
    `test` after training and logs those metrics.
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

import mlflow
import mlflow.keras
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.random import set_seed

from forex_ml.config import TrainParams, load_params
from forex_ml.data.splitting import Splits
from forex_ml.evaluation.baselines import majority_class_baseline, persistence_baseline
from forex_ml.evaluation.class_balance import class_balance
from forex_ml.evaluation.multiple_comparisons import config_signature_from_params
from forex_ml.paths import pair_key, splits_npz_path
from forex_ml.training.model import build_lstm_regressor, compile_model, configure_gpu_memory_growth


def _build_callbacks(params: TrainParams, checkpoint_path: Path) -> list:
    return [
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=params.reduce_lr_on_plateau_factor,
            patience=params.reduce_lr_on_plateau_patience,
        ),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=params.early_stopping_patience,
            restore_best_weights=True,
        ),
    ]


def _log_history(history_dict: dict) -> None:
    metric_names = [k for k in history_dict if not k.startswith("val_")]
    for metric_name in metric_names:
        for epoch, value in enumerate(history_dict[metric_name]):
            mlflow.log_metric(f"train_{metric_name}", value, step=epoch)
        val_key = f"val_{metric_name}"
        if val_key in history_dict:
            for epoch, value in enumerate(history_dict[val_key]):
                mlflow.log_metric(val_key, value, step=epoch)


def train_and_evaluate(
    splits: Splits,
    params: TrainParams,
    instrument: str,
    granularity: str,
    output_dir: Path,
    n_back: int,
    lookahead: int,
    *,
    experiment_name: str | None = None,
    register_model: bool = True,
    extra_params: dict | None = None,
    run_name_suffix: str | None = None,
) -> dict:
    """Train on `splits.train`, validate on the real Stage-2 `splits.val`, evaluate on
    the held-out `splits.test`, and log params/metrics/model to MLflow. Returns the
    dict of test metrics.

    `n_back`/`lookahead` are FeatureParams, not TrainParams, but are logged here
    anyway (not just used to locate the splits file) so that
    forex_ml.evaluation.multiple_comparisons's config-signature hash can tell two
    runs with different windowing apart. Without this, two runs that only differ in
    n_back would log an IDENTICAL set of params (TrainParams is unchanged) and get
    silently collapsed into "the same configuration, just retrained" -- keeping only
    the most recent one and discarding the other, exactly the failure mode that
    whole-config hashing was built to prevent.

    `experiment_name`/`register_model`/`extra_params`/`run_name_suffix` exist for
    forex_ml.evaluation.rolling_cv, which trains many folds of the SAME
    configuration across different time windows purely as a robustness diagnostic —
    those runs should log to their own MLflow experiment (not the pair's normal one,
    which forex_ml.evaluation.multiple_comparisons scans for one "official" run per
    (pair, configuration)) and should NOT register a model (they aren't deployment
    candidates). Defaults reproduce the original single-split behavior exactly.
    """
    configure_gpu_memory_growth()
    set_seed(params.tensorflow_seed)

    run_uid = str(uuid.uuid4())
    model_dir = Path(output_dir) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / f"{run_uid}_checkpoint.keras"

    input_shape = (splits.train["M"].shape[1], splits.train["M"].shape[2])
    num_outputs = splits.train["y"].shape[1]

    mlflow.set_tracking_uri(params.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name or params.mlflow_experiment_name)

    run_name = f"{instrument}_{granularity}_{run_uid}"
    if run_name_suffix:
        run_name = f"{run_name}_{run_name_suffix}"

    with mlflow.start_run(run_name=run_name) as run:
        logged_params = {
            "instrument": instrument,
            "granularity": granularity,
            "run_uid": run_uid,
            "n_back": n_back,
            "lookahead": lookahead,
            **params.model_dump(exclude={"mlflow_experiment_name", "mlflow_tracking_uri"}),
            **(extra_params or {}),
        }
        mlflow.log_params(logged_params)

        # Cheap regime-drift check, logged before training even starts: train's class
        # balance is close to even by construction (thresholds come from train
        # quantiles), but val/test aren't guaranteed to be if the volatility regime
        # has shifted between periods.
        for split_name, split in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
            mlflow.log_metrics({
                f"{split_name}_{class_name}_balance": fraction
                for class_name, fraction in class_balance(split["y"]).items()
            })

        model = build_lstm_regressor(params, input_shape, num_outputs)
        compile_model(model, params)

        history = model.fit(
            splits.train["M"], splits.train["y"],
            validation_data=(splits.val["M"], splits.val["y"]),
            epochs=params.epochs,
            batch_size=params.batch_size,
            callbacks=_build_callbacks(params, checkpoint_path),
        )
        _log_history(history.history)

        test_results = model.evaluate(splits.test["M"], splits.test["y"], return_dict=True)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_results.items()})

        # Baselines, logged alongside the LSTM's own test metrics so they're directly
        # comparable in the MLflow UI without cross-referencing separate runs. Neither
        # baseline uses the model or X — if test_accuracy doesn't clear these, the
        # LSTM isn't adding value over a trivial rule.
        majority_result = majority_class_baseline(splits.train["y"], splits.test["y"])
        persistence_result = persistence_baseline(splits.test["y"])
        test_results["baseline_majority_accuracy"] = majority_result["accuracy"]
        test_results["baseline_persistence_accuracy"] = persistence_result["accuracy"]
        mlflow.log_metrics({
            "baseline_majority_test_accuracy": majority_result["accuracy"],
            "baseline_persistence_test_accuracy": persistence_result["accuracy"],
        })

        # Per-row correctness (not just aggregate accuracy), saved as an artifact so a
        # proper paired significance test (McNemar's — see
        # forex_ml/evaluation/multiple_comparisons.py) can compare the LSTM against a
        # baseline on the SAME test rows, rather than treating them as independent
        # samples. lstm_correct and majority_correct are aligned 1:1 with the full
        # test set; persistence_correct is one row shorter (see persistence_baseline).
        #
        # Also saved: the raw softmax probabilities (not just top-1 correctness) and
        # the test row's timestamp/price/spread/y_raw from splits.test (see
        # forex_ml.data.splitting.Splits) -- a real backtest (forex-strategy) needs
        # "how confident was the model, at what price, at what spread cost, and what
        # was the actual realized outcome" per row, which a correct/incorrect boolean
        # alone can't answer.
        lstm_pred_proba = model.predict(splits.test["M"], verbose=0)
        lstm_pred_idx = np.argmax(lstm_pred_proba, axis=1)
        lstm_true_idx = np.argmax(splits.test["y"], axis=1)
        predictions_path = model_dir / f"{run_uid}_predictions.npz"
        np.savez_compressed(
            predictions_path,
            lstm_correct=(lstm_pred_idx == lstm_true_idx),
            majority_correct=majority_result["correct"],
            persistence_correct=persistence_result["correct"],
            lstm_pred_proba=lstm_pred_proba,
            test_timestamp=splits.test["timestamp"],
            test_price=splits.test["price"],
            test_spread=splits.test["spread"],
            test_y_raw=splits.test["y_raw"],
        )
        mlflow.log_artifact(str(predictions_path))

        mlflow.keras.log_model(model, name="model")
        if register_model:
            # Tag the registered model version with instrument/granularity/config
            # signature (the same hash forex_ml.evaluation.multiple_comparisons uses
            # to group runs) so forex-strategy can look up "the model for
            # (instrument, granularity, config)" directly via MlflowClient, instead of
            # grepping every version's source run params -- the registry itself has
            # no per-pair identity otherwise (every pair registers under the same
            # shared `params.mlflow_experiment_name`).
            config_sig = config_signature_from_params(logged_params)
            mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/model",
                name=params.mlflow_experiment_name,
                tags={"instrument": instrument, "granularity": granularity, "config_signature": config_sig},
            )

        history_path = model_dir / f"{run_uid}_history.json"
        history_path.write_text(json.dumps(history.history, indent=2))
        mlflow.log_artifact(str(history_path))

    return test_results


def run(instrument: str, granularity: str, params_path: str | Path | None = None) -> dict:
    """CLI/DVC/Prefect entry point: load params + Stage-2 splits for one pair, train,
    evaluate, log to MLflow. Returns test metrics."""
    params = load_params(params_path) if params_path else load_params()
    key = pair_key(instrument, granularity, params.feature.n_back, params.feature.lookahead)
    splits = Splits.load_npz(splits_npz_path(params.feature.output_dir, key))
    return train_and_evaluate(
        splits, params.train, instrument, granularity, Path(params.feature.output_dir),
        params.feature.n_back, params.feature.lookahead,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the LSTM regressor for one (instrument, granularity) pair.")
    parser.add_argument("--instrument", required=True, help="e.g. EUR/USD")
    parser.add_argument("--granularity", required=True, help="e.g. H1")
    parser.add_argument("--params", default=None, help="Path to params.yaml (default: repo root)")
    args = parser.parse_args()

    test_results = run(args.instrument, args.granularity, args.params)
    print(json.dumps(test_results, indent=2))


if __name__ == "__main__":
    main()
