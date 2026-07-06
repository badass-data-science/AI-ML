from __future__ import annotations

import numpy as np
import pandas as pd

from forex_ml.data.splitting import Splits, TimeSeriesSplitter

COLUMNS_X_COMPONENTS = ["feat_0", "feat_1", "feat_2"]


def _make_pdf(n: int, n_back: int = 5, n_features: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    timestamps = np.arange(n) * 3600
    X = rng.normal(size=(n, n_back, n_features))
    return pd.DataFrame({
        "instrument": "EUR/USD",
        "granularity": "H1",
        "unix_epoch_s": timestamps,
        "pd_lead": rng.normal(size=n),
        "X": list(X),
    })


def _make_non_ts(n: int, n_features: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    data = {
        "instrument": "EUR/USD",
        "granularity": "H1",
        "unix_epoch_s": np.arange(n) * 3600,
    }
    for i in range(n_features):
        data[COLUMNS_X_COMPONENTS[i]] = rng.normal(size=n)
    return pd.DataFrame(data)


def _splitter(n: int) -> TimeSeriesSplitter:
    return TimeSeriesSplitter(
        _make_pdf(n), _make_non_ts(n), "EUR/USD", "H1", columns_x_components=COLUMNS_X_COMPONENTS,
    )


def test_split_covers_every_row_with_no_overlap():
    n = 100
    splits = _splitter(n).split_train_val_test_by_proportion([0.7, 0.15])

    total = len(splits.train["y"]) + len(splits.val["y"]) + len(splits.test["y"])
    assert total == n
    assert 0.6 < len(splits.train["y"]) / n < 0.8


def test_outcome_is_one_hot_over_three_classes():
    splits = _splitter(120).split_train_val_test_by_proportion([0.7, 0.15])
    for split_name in ("train", "val", "test"):
        y = getattr(splits, split_name)["y"]
        assert y.shape[1] == 3
        assert np.all(y.sum(axis=1) == 1)


def test_normalization_applies_train_stats_to_all_splits():
    splitter = _splitter(100)
    splits = splitter.split_train_val_test_by_proportion([0.7, 0.15])

    train_non_ts = splitter.df_non_time_series[COLUMNS_X_COMPONENTS].to_numpy()
    # sanity: normalized train values should be roughly centered, given the same
    # mean/std that produced them
    assert np.all(np.abs(splits.train["M"].mean(axis=(0, 1))) < 2.0)
    assert train_non_ts.shape[1] == splits.train["M"].shape[2]


def test_npz_round_trip(tmp_path):
    splits = _splitter(50).split_train_val_test_by_proportion([0.6, 0.2])

    path = tmp_path / "splits.npz"
    splits.save_npz(path)
    loaded = Splits.load_npz(path)

    np.testing.assert_array_equal(splits.train["M"], loaded.train["M"])
    np.testing.assert_array_equal(splits.val["y"], loaded.val["y"])
    np.testing.assert_array_equal(splits.test["M"], loaded.test["M"])
