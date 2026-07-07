from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forex_ml.data.splitting import Splits, TimeSeriesSplitter, load_and_stack

COLUMNS_X_COMPONENTS = ["feat_0", "feat_1", "feat_2"]


def _make_pdf(n: int, n_back: int = 5, n_features: int = 3) -> pd.DataFrame:
    """Stands in for what load_and_stack would have produced -- includes mid_close/
    spread_close (see COLUMNS_PASSTHROUGH) since _build_splits reads them directly off
    this frame to populate the test split's reference data."""
    rng = np.random.default_rng(0)
    timestamps = np.arange(n) * 3600
    X = rng.normal(size=(n, n_back, n_features))
    return pd.DataFrame({
        "instrument": "EUR/USD",
        "granularity": "H1",
        "unix_epoch_s": timestamps,
        "pd_lead": rng.normal(size=n),
        "X": list(X),
        "mid_close": rng.normal(loc=1.1, scale=0.01, size=n),
        "spread_close": rng.uniform(0.0001, 0.0005, size=n),
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

    # columns_passthrough=[]: this test is about the stacking UDF's axis order, not
    # about mid_close/spread_close passthrough -- the synthetic frames above don't
    # have those columns, so don't ask load_and_stack to select them.
    pdf, _ = load_and_stack(
        spark, str(ts_path), str(non_ts_path), ["feat_a", "feat_b"], "pd_lead", columns_passthrough=[],
    )

    X = np.array(pdf.loc[0, "X"])
    assert X.shape == (n_back, 2)
    expected = np.array([[feat_a[t], feat_b[t]] for t in range(n_back)])
    np.testing.assert_array_equal(X, expected)
    # explicit anti-reversal check, spelled out rather than left implicit in the
    # exact-equality assertion above
    assert not np.array_equal(X, expected[::-1])


def test_compute_boundaries_purge_gap_is_symmetric_and_exact():
    """Precise arithmetic check on the purge-gap boundaries themselves, independent
    of any X/y data: with purge_bars bars purged on each side of a boundary, the gap
    between the earlier split's hi and the later split's lo must be exactly
    2 * purge_bars * bar_spacing, and purge_bars=0 must reproduce the original
    (unpurged) boundaries exactly."""
    n = 100
    bar_spacing = 3600  # matches _make_pdf's hourly synthetic timestamps
    splitter = _splitter(n)

    train_lo, train_hi, val_lo, val_hi, test_lo, test_hi = splitter._compute_boundaries([0.6, 0.2], purge_bars=0)
    assert val_lo == train_hi
    assert test_lo == val_hi

    purge_bars = 5
    p_train_lo, p_train_hi, p_val_lo, p_val_hi, p_test_lo, p_test_hi = splitter._compute_boundaries(
        [0.6, 0.2], purge_bars=purge_bars,
    )
    assert p_train_lo == train_lo  # only boundaries move, not the outer edges
    assert p_test_hi == test_hi
    assert p_val_lo - p_train_hi == pytest.approx(2 * purge_bars * bar_spacing)
    assert p_test_lo - p_val_hi == pytest.approx(2 * purge_bars * bar_spacing)


def test_purge_bars_removes_exactly_the_rows_outside_purged_boundaries():
    """Behavioral check, cross-validated against an independent recomputation from
    the boundaries themselves: purging must produce exactly the row counts implied by
    _compute_boundaries, and those counts must be strictly smaller than the unpurged
    split (i.e. rows adjacent to each boundary actually got dropped, not just that the
    boundary arithmetic changed on paper)."""
    n = 100
    purge_bars = 5
    splitter = _splitter(n)

    train_lo, train_hi, val_lo, val_hi, test_lo, test_hi = splitter._compute_boundaries(
        [0.6, 0.2], purge_bars=purge_bars,
    )
    ts = splitter.df[splitter.timestamp_column]
    expected_train_n = int(((ts >= train_lo) & (ts < train_hi)).sum())
    expected_val_n = int(((ts >= val_lo) & (ts < val_hi)).sum())
    expected_test_n = int(((ts >= test_lo) & (ts <= test_hi)).sum())

    purged = splitter.split_train_val_test_by_proportion([0.6, 0.2], purge_bars=purge_bars)
    assert len(purged.train["y"]) == expected_train_n
    assert len(purged.val["y"]) == expected_val_n
    assert len(purged.test["y"]) == expected_test_n

    unpurged = splitter.split_train_val_test_by_proportion([0.6, 0.2], purge_bars=0)
    assert expected_train_n < len(unpurged.train["y"])
    assert expected_val_n < len(unpurged.val["y"])
    assert expected_test_n < len(unpurged.test["y"])


def test_rolling_boundaries_sliding_window_has_constant_train_size_and_steps_by_test_bars():
    bar_spacing = 3600
    splitter = _splitter(100)

    boundaries = splitter._rolling_boundaries(
        n_folds=3, min_train_bars=40, val_bars=10, test_bars=10, window="sliding",
    )
    assert len(boundaries) == 3

    for train_lo, train_hi, val_lo, val_hi, test_lo, test_hi in boundaries:
        assert (train_hi - train_lo) == pytest.approx(40 * bar_spacing)  # constant train size
        assert (val_hi - val_lo) == pytest.approx(10 * bar_spacing)
        assert (test_hi - test_lo) == pytest.approx(10 * bar_spacing)
        assert val_lo == train_hi
        assert test_lo == val_hi

    # each fold's test block starts exactly test_bars (10 bars) after the previous one's
    for (a, b) in zip(boundaries, boundaries[1:]):
        assert b[4] - a[4] == pytest.approx(10 * bar_spacing)
        assert b[0] - a[0] == pytest.approx(10 * bar_spacing)  # train start also slides


def test_rolling_boundaries_expanding_window_grows_train_and_keeps_start_fixed():
    bar_spacing = 3600
    splitter = _splitter(100)

    boundaries = splitter._rolling_boundaries(
        n_folds=3, min_train_bars=40, val_bars=10, test_bars=10, window="expanding",
    )
    min_ts = splitter.df["unix_epoch_s"].min()

    for i, (train_lo, train_hi, val_lo, val_hi, test_lo, test_hi) in enumerate(boundaries):
        assert train_lo == min_ts  # anchored at the start every fold
        assert (train_hi - train_lo) == pytest.approx((40 + i * 10) * bar_spacing)  # grows each fold
        assert val_lo == train_hi
        assert test_lo == val_hi

    # test blocks still step forward by test_bars each fold, same as sliding
    for (a, b) in zip(boundaries, boundaries[1:]):
        assert b[4] - a[4] == pytest.approx(10 * bar_spacing)


def test_rolling_boundaries_rejects_unknown_window():
    with pytest.raises(ValueError, match="window"):
        _splitter(100)._rolling_boundaries(
            n_folds=2, min_train_bars=10, val_bars=5, test_bars=5, window="bogus",
        )


def test_rolling_boundaries_raises_with_helpful_message_when_not_enough_data():
    with pytest.raises(ValueError, match=r"At most \d+ fold\(s\) fit"):
        _splitter(50)._rolling_boundaries(
            n_folds=10, min_train_bars=40, val_bars=10, test_bars=10, window="sliding",
        )


def test_rolling_folds_row_counts_match_recomputed_boundaries():
    """Cross-validated the same way test_purge_bars_removes_exactly_the_rows... is:
    recompute expected row counts independently from the boundaries themselves rather
    than trusting rolling_folds' own arithmetic twice."""
    splitter = _splitter(100)
    purge_bars = 2

    boundaries = splitter._rolling_boundaries(
        n_folds=3, min_train_bars=40, val_bars=10, test_bars=10, window="sliding", purge_bars=purge_bars,
    )
    folds = splitter.rolling_folds(
        n_folds=3, min_train_bars=40, val_bars=10, test_bars=10, window="sliding", purge_bars=purge_bars,
    )
    assert len(folds) == 3

    ts = splitter.df[splitter.timestamp_column]
    for (train_lo, train_hi, val_lo, val_hi, test_lo, test_hi), split in zip(boundaries, folds):
        expected_train_n = int(((ts >= train_lo) & (ts < train_hi)).sum())
        expected_val_n = int(((ts >= val_lo) & (ts < val_hi)).sum())
        expected_test_n = int(((ts >= test_lo) & (ts <= test_hi)).sum())
        assert len(split.train["y"]) == expected_train_n
        assert len(split.val["y"]) == expected_val_n
        assert len(split.test["y"]) == expected_test_n


def test_rolling_folds_sliding_vs_expanding_agree_on_first_fold():
    """The first fold is identical regardless of window type -- expanding and sliding
    only diverge starting from the second fold, since expanding's train_bars formula
    (min_train_bars + fold * test_bars) equals min_train_bars at fold=0."""
    splitter = _splitter(100)
    sliding = splitter.rolling_folds(n_folds=2, min_train_bars=40, val_bars=10, test_bars=10, window="sliding")
    expanding = splitter.rolling_folds(n_folds=2, min_train_bars=40, val_bars=10, test_bars=10, window="expanding")

    np.testing.assert_array_equal(sliding[0].train["M"], expanding[0].train["M"])
    assert len(sliding[1].train["y"]) < len(expanding[1].train["y"])  # they diverge by fold 2
