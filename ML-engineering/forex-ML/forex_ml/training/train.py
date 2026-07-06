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
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.random import set_seed

from forex_ml.config import TrainParams, load_params
from forex_ml.data.splitting import Splits
from forex_ml.evaluation.baselines import majority_class_baseline, persistence_baseline
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
) -> dict:
    """Train on `splits.train`, validate on the real Stage-2 `splits.val`, evaluate on
    the held-out `splits.test`, and log params/metrics/model to MLflow. Returns the
    dict of test metrics."""
    configure_gpu_memory_growth()
    set_seed(params.tensorflow_seed)

    run_uid = str(uuid.uuid4())
    model_dir = Path(output_dir) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / f"{run_uid}_checkpoint.keras"

    input_shape = (splits.train["M"].shape[1], splits.train["M"].shape[2])
    num_outputs = splits.train["y"].shape[1]

    mlflow.set_tracking_uri(params.mlflow_tracking_uri)
    mlflow.set_experiment(params.mlflow_experiment_name)

    with mlflow.start_run(run_name=f"{instrument}_{granularity}_{run_uid}"):
        mlflow.log_params({
            "instrument": instrument,
            "granularity": granularity,
            "run_uid": run_uid,
            **params.model_dump(exclude={"mlflow_experiment_name", "mlflow_tracking_uri"}),
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

        mlflow.keras.log_model(model, name="model", registered_model_name=params.mlflow_experiment_name)

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
    return train_and_evaluate(splits, params.train, instrument, granularity, Path(params.feature.output_dir))


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
