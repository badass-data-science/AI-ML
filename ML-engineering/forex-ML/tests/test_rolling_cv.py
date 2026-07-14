from __future__ import annotations

import mlflow
import yaml

from forex_ml.config import DEFAULT_PARAMS_PATH, load_params
from forex_ml.evaluation.rolling_cv import run_rolling_cv
from forex_ml.flows.prepare_data_flow import engineer_and_save_task


def _write_small_params(tmp_path, output_dir, tracking_uri, experiment_name: str) -> str:
    raw = yaml.safe_load(DEFAULT_PARAMS_PATH.read_text(encoding="utf-8"))
    raw["feature"].update({
        "instruments": ["EUR/USD"],
        "granularities": ["H1"],
        "n_back": 10,
        "lookahead": 2,
        "ma_lookback_list": [3, 5],
        "min_training_timestamp": "2020-01-01T00:00:00",
        "output_dir": str(output_dir),
    })
    # return_MA_96/volatility_MA_96/return_zscore_12/rsi_12 (production columns_x)
    # don't exist under this scaled-down ma_lookback_list=[3, 5] (min/max lookback
    # is 3/5 here, not 12/96) -- drop them here rather than scaling ma_lookback_list
    # up, which would need far more synthetic candle rows. usd_strength_return also
    # dropped: this test's engineer_and_save_task call doesn't set up cross-pair data.
    raw["split"]["columns_x"] = [
        c for c in raw["split"]["columns_x"]
        if c not in ("return_MA_96", "volatility_MA_96", "return_zscore_12", "rsi_12", "usd_strength_return")
    ]
    raw["train"].update({
        "number_of_cells_per_rnn_layer": [4],
        "number_of_cells_per_dense_layer": [4],
        "epochs": 1,
        "batch_size": 16,
        "mlflow_experiment_name": experiment_name,
        "mlflow_tracking_uri": tracking_uri,
    })
    path = tmp_path / "small_params.yaml"
    path.write_text(yaml.dump(raw))
    return str(path)


def test_run_rolling_cv_end_to_end(spark, synthetic_candles, tmp_path):
    """Real end-to-end: engineer Stage-1 features, run 2 sliding-window rolling
    folds, and verify each fold trained + logged to its OWN experiment (not the
    pair's normal one) without registering a model -- the whole point of this
    being a robustness diagnostic rather than a deployment strategy."""
    output_dir = tmp_path / "output"
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    experiment_name = "rolling-cv-test"

    params_path = _write_small_params(tmp_path, output_dir, tracking_uri, experiment_name)
    feature_params = load_params(params_path).feature
    engineer_and_save_task(spark, synthetic_candles, "EUR/USD", "H1", feature_params)

    report = run_rolling_cv(
        spark, "EUR/USD", "H1",
        n_folds=2, min_train_bars=100, val_bars=40, test_bars=40, purge_bars=5,
        window="sliding", params_path=params_path,
    )

    assert report["n_folds"] == 2
    assert report["window"] == "sliding"
    assert len(report["fold_lstm_scores"]) == 2
    assert len(report["fold_baseline_majority_scores"]) == 2
    assert len(report["fold_baseline_persistence_scores"]) == 2
    for stats_key in ("lstm", "baseline_majority", "baseline_persistence"):
        assert set(report[stats_key]) == {"mean", "std", "min", "max"}

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    rolling_experiment = client.get_experiment_by_name(f"{experiment_name}-rolling-cv")
    assert rolling_experiment is not None
    runs = client.search_runs([rolling_experiment.experiment_id])
    assert len(runs) == 2
    fold_indices = set()
    for run in runs:
        assert run.data.params["diagnostic"] == "rolling_cv"
        assert run.data.params["window_type"] == "sliding"
        fold_indices.add(run.data.params["fold_index"])
    assert fold_indices == {"0", "1"}

    # the pair's normal experiment was never touched
    assert client.get_experiment_by_name(experiment_name) is None

    # no model registered anywhere -- these are diagnostic runs, not deployment candidates
    assert len(client.search_registered_models()) == 0


def test_run_rolling_cv_expanding_window(spark, synthetic_candles, tmp_path):
    """Same end-to-end path with the expanding window, checked separately since
    fold train-set sizes differ (see TimeSeriesSplitter.rolling_folds)."""
    output_dir = tmp_path / "output"
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    experiment_name = "rolling-cv-expanding-test"

    params_path = _write_small_params(tmp_path, output_dir, tracking_uri, experiment_name)
    feature_params = load_params(params_path).feature
    engineer_and_save_task(spark, synthetic_candles, "EUR/USD", "H1", feature_params)

    report = run_rolling_cv(
        spark, "EUR/USD", "H1",
        n_folds=2, min_train_bars=100, val_bars=40, test_bars=40, purge_bars=5,
        window="expanding", params_path=params_path,
    )

    assert report["window"] == "expanding"
    assert len(report["fold_lstm_scores"]) == 2

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    rolling_experiment = client.get_experiment_by_name(f"{experiment_name}-rolling-cv")
    runs = client.search_runs([rolling_experiment.experiment_id])
    assert all(run.data.params["window_type"] == "expanding" for run in runs)
