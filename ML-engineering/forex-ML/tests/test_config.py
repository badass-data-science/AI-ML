from __future__ import annotations

import pandas as pd
import yaml
import pytest

from forex_ml.config import DEFAULT_PARAMS_PATH, load_params
from forex_ml.data.features import engineer_features


def test_engineered_columns_matches_actual_stage1_output(spark, synthetic_candles):
    """FeatureParams.engineered_columns is a hand-maintained mirror of whatever
    forex_ml.data.features.engineer_features actually produces -- PipelineParams
    only checks that split.columns_x is a SUBSET of this mirror, never that the
    mirror itself is accurate. This closes that gap: run the real Stage-1 pipeline
    (against synthetic candles, same fixture test_features.py uses) with the
    default params.yaml feature config and assert the two sets are exactly equal,
    so a new transform's output columns can't silently drift from this mirror the
    way ma_lookback_list/ma_columns_list-derived columns already could before.

    Passes a synthetic cross_pair_usd_strength too, matching what prepare_data_flow.py
    always assembles in production -- otherwise "usd_strength_return" would be in
    the mirror but never actually produced by this call, a false-positive drift.
    """
    params = load_params()
    df = spark.createDataFrame(synthetic_candles)
    cross_pair_usd_strength = spark.createDataFrame(pd.DataFrame({
        "unix_epoch_s": synthetic_candles["unix_epoch_s"],
        "usd_strength_return": 0.0,
    }))
    _, _, columns_x = engineer_features(
        df,
        ma_lookback_list=params.feature.ma_lookback_list,
        ma_columns_list=params.feature.ma_columns_list,
        columns_base=params.feature.columns_base,
        lookahead=params.feature.lookahead,
        n_back=params.feature.n_back,
        training_and_testing=params.feature.training_and_testing,
        cross_pair_usd_strength=cross_pair_usd_strength,
    )
    assert set(columns_x) == params.feature.engineered_columns


def test_default_params_yaml_loads():
    params = load_params()
    assert params.feature.n_back == 200
    assert "EUR/USD" in params.feature.instruments
    assert params.train.epochs > 0


def test_split_columns_x_must_exist_in_engineered_features(tmp_path):
    raw = yaml.safe_load(DEFAULT_PARAMS_PATH.read_text(encoding="utf-8"))
    raw["split"]["columns_x"] = ["not_a_real_column"]
    bad_path = tmp_path / "bad_params.yaml"
    bad_path.write_text(yaml.dump(raw))

    with pytest.raises(ValueError, match="not_a_real_column"):
        load_params(bad_path)


def test_train_val_proportion_must_leave_room_for_test(tmp_path):
    raw = yaml.safe_load(DEFAULT_PARAMS_PATH.read_text(encoding="utf-8"))
    raw["split"]["train_val_proportion"] = [0.7, 0.35]
    bad_path = tmp_path / "bad_params.yaml"
    bad_path.write_text(yaml.dump(raw))

    with pytest.raises(ValueError, match="sum to < 1.0"):
        load_params(bad_path)


def test_train_val_proportion_needs_exactly_two_entries(tmp_path):
    raw = yaml.safe_load(DEFAULT_PARAMS_PATH.read_text(encoding="utf-8"))
    raw["split"]["train_val_proportion"] = [0.7]
    bad_path = tmp_path / "bad_params.yaml"
    bad_path.write_text(yaml.dump(raw))

    with pytest.raises(ValueError, match="exactly 2 entries"):
        load_params(bad_path)
