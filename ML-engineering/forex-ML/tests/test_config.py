from __future__ import annotations

import yaml
import pytest

from forex_ml.config import DEFAULT_PARAMS_PATH, load_params


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
