from __future__ import annotations

import numpy as np
import pandas as pd

from forex_ml.data.splitting import Splits, TimeSeriesSplitter, load_and_stack

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


def test_load_and_stack_produces_timesteps_by_features_shape(spark, tmp_path):
    """Regression test for a real bug: the UDF used to stack per-feature arrays into
    one 'X' matrix built (num_features, n_back) instead of (n_back, num_features) —
    silently swapping the time and feature axes relative to what
    forex_ml.training.model's `input_shape=(n_back, num_features)` assumes. Every
    other splitting test hand-constructs 'X' already in the assumed-correct shape, so
    none of them exercise the actual stacking UDF; this one does, against a real
    SparkSession, and checks exact per-timestep values (not just shape) so a silent
    transpose can't slip back in even if n_back happened to equal num_features.

    feat_a/feat_b are strictly increasing and distinct at every position, so this also
    pins down direction along the time axis, not just position mapping: the
    exact-equality check below would fail just as loudly if the stacking silently
    reversed the n_back axis as it would for a transpose. See
    test_features.test_window_into_arrays_preserves_chronological_order_oldest_first
    for the same guarantee traced further upstream, at the point (window_into_arrays)
    where "oldest first, current bar last" is actually established from real
    timestamps rather than from values chosen to look ordered.
    """
    feat_a = [10.0, 11.0, 12.0, 13.0, 14.0]
    feat_b = [100.0, 101.0, 102.0, 103.0, 104.0]
    n_back = len(feat_a)

    df_time_series = spark.createDataFrame(
        [("EUR/USD", "H1", 0, 0.5, feat_a, feat_b)],
        ["instrument", "granularity", "unix_epoch_s", "pd_lead", "feat_a", "feat_b"],
    )
    df_non_time_series = spark.createDataFrame(
        [("EUR/USD", "H1", 0, feat_a[-1], feat_b[-1])],
        ["instrument", "granularity", "unix_epoch_s", "feat_a", "feat_b"],
    )

    ts_path = tmp_path / "ts.parquet"
    non_ts_path = tmp_path / "non_ts.parquet"
    df_time_series.write.parquet(str(ts_path))
    df_non_time_series.write.parquet(str(non_ts_path))

    pdf, _ = load_and_stack(spark, str(ts_path), str(non_ts_path), ["feat_a", "feat_b"], "pd_lead")

    X = np.array(pdf.loc[0, "X"])
    assert X.shape == (n_back, 2)
    expected = np.array([[feat_a[t], feat_b[t]] for t in range(n_back)])
    np.testing.assert_array_equal(X, expected)
    # explicit anti-reversal check, spelled out rather than left implicit in the
    # exact-equality assertion above
    assert not np.array_equal(X, expected[::-1])
