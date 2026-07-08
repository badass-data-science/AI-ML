"""Locate and load a trained forex-ML model from its MLflow registry.

Every (instrument, granularity, config) trains under the SAME registered-model name
(forex-ML's `train.mlflow_experiment_name`, e.g. "forex-lstm") -- the registry itself
has no per-pair identity. forex_ml.training.train tags each version at registration
time with instrument/granularity/config_signature (see forex-ML's README, "Finding
'the model for (instrument, granularity)'"), which is what this module searches on.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlflow.keras
import numpy as np
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient


@dataclass
class ResolvedModel:
    version: ModelVersion
    run_id: str
    instrument: str
    granularity: str
    config_signature: str
    column_y: str


def find_model_version(
    client: MlflowClient,
    registered_model_name: str,
    instrument: str,
    granularity: str,
    config_signature: str | None = None,
    column_y: str | None = None,
) -> ResolvedModel:
    """Most recently registered version tagged for this (instrument, granularity
    [, config_signature] [, column_y]). Raises ValueError if none match -- there's
    no sensible fallback (e.g. "just pick any version") for a caller asking for a
    specific pair.

    `column_y` matters beyond convenience: forex-ML's production target is now
    always `"triple_barrier"`, but older, pre-migration model versions may still be
    registered for the same (instrument, granularity), tagged `pd_lead`/
    `volatility_lead` or untagged entirely -- filtering on `column_y="triple_barrier"`
    (see run_backtest.py) is what excludes those from a search that expects the
    current scheme, without grepping every candidate version's source run.
    """
    filter_parts = [
        f"name = '{registered_model_name}'",
        f"tags.instrument = '{instrument}'",
        f"tags.granularity = '{granularity}'",
    ]
    if config_signature is not None:
        filter_parts.append(f"tags.config_signature = '{config_signature}'")
    if column_y is not None:
        filter_parts.append(f"tags.column_y = '{column_y}'")

    versions = client.search_model_versions(" and ".join(filter_parts))
    if not versions:
        suffix = "".join([
            f" (config_signature={config_signature})" if config_signature else "",
            f" (column_y={column_y})" if column_y else "",
        ])
        raise ValueError(
            f"No registered version of {registered_model_name!r} tagged for {instrument} {granularity}{suffix}"
        )

    latest = max(versions, key=lambda v: int(v.version))
    assert latest.run_id is not None  # every registered version is created from a run
    return ResolvedModel(
        version=latest,
        run_id=latest.run_id,
        instrument=latest.tags.get("instrument", instrument),
        granularity=latest.tags.get("granularity", granularity),
        config_signature=latest.tags.get("config_signature", ""),
        column_y=latest.tags.get("column_y", ""),
    )


def load_keras_model(resolved: ResolvedModel):
    """The actual Keras model object, ready for `.predict()` -- a thin wrapper so
    callers never have to build the `models:/name/version` URI by hand."""
    return mlflow.keras.load_model(f"models:/{resolved.version.name}/{resolved.version.version}")


def load_test_predictions(client: MlflowClient, run_id: str, download_dir: str) -> dict[str, np.ndarray]:
    """Download and load the `<run_uid>_predictions.npz` artifact forex_ml.training.train
    logs for this run: lstm_pred_proba/test_timestamp/test_price/test_spread/
    test_y_raw/test_exit_bar_offset/test_realized_volatility/lstm_correct/
    majority_correct/persistence_correct."""
    artifact_path = next(a.path for a in client.list_artifacts(run_id) if a.path.endswith("_predictions.npz"))
    local_path = client.download_artifacts(run_id, artifact_path, download_dir)
    return dict(np.load(local_path))
