from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import pytest

from forex_ml.data.features import (
    add_calendar_features,
    add_cross_pair_features,
    add_market_features,
    add_session_features,
    add_targets,
    compute_cross_pair_usd_strength,
    engineer_features,
    window_into_arrays,
)


def test_add_calendar_features_adds_expected_columns(spark, synthetic_candles):
    df = spark.createDataFrame(synthetic_candles)
    df = add_calendar_features(df)
    for col in ["day_sin", "day_cos", "week_sin", "week_cos"]:
        assert col in df.columns
    row = df.orderBy("unix_epoch_s").first()
    assert -1.0 <= row["day_sin"] <= 1.0
    assert -1.0 <= row["day_cos"] <= 1.0


def test_add_market_features_computes_volatility_and_lag_diffs(spark, synthetic_candles):
    df = spark.createDataFrame(synthetic_candles)
    df = add_market_features(df)
    pdf = df.orderBy("unix_epoch_s").toPandas()

    # high - low is always >= 0 by construction of the synthetic candles
    assert (pdf["volatility"] >= 0).all()
    # diff_spread_close/diff_volume are lags: null on the very first row, present after
    assert pd.isna(pdf.loc[0, "diff_spread_close"])
    assert not pd.isna(pdf.loc[1, "diff_spread_close"])


def test_add_market_features_realized_volatility_is_a_fixed_12_bar_trailing_average(spark, synthetic_candles):
    """Independent of ma_lookback_list/ma_columns_list -- must exist regardless of
    whatever moving-average windows are configured for actual model features (every
    test in this suite uses ma_lookback_list=[3, 5]; production uses [12, 30, 50])."""
    df = spark.createDataFrame(synthetic_candles)
    df = add_market_features(df)
    pdf = df.orderBy("unix_epoch_s").toPandas()

    manual = pdf["volatility"].rolling(window=12, min_periods=1).mean()
    np.testing.assert_allclose(pdf["realized_volatility"].to_numpy(), manual.to_numpy(), rtol=1e-4)


def test_compute_cross_pair_usd_strength_averages_sign_adjusted_returns(spark):
    """USD/JPY is USD-base: a positive return means USD strengthened, contributes
    directly. EUR/USD is USD-quote: a NEGATIVE return also means USD strengthened
    (EUR weakened), so it must contribute with a flipped sign -- both pairs here
    agree "USD strengthened", so the averaged signal should come out positive, not
    cancel out or come out negative from a sign-convention mistake."""
    ts = [1000, 2000]
    usdjpy = spark.createDataFrame(pd.DataFrame({"unix_epoch_s": ts, "return": [0.01, 0.01]}))
    eurusd = spark.createDataFrame(pd.DataFrame({"unix_epoch_s": ts, "return": [-0.02, -0.02]}))

    result = (
        compute_cross_pair_usd_strength({"USD/JPY": usdjpy, "EUR/USD": eurusd})
        .orderBy("unix_epoch_s").toPandas()
    )

    # signed: USD/JPY contributes +0.01, EUR/USD contributes -(-0.02) = +0.02 -> avg 0.015
    np.testing.assert_allclose(result["usd_strength_return"].to_numpy(), [0.015, 0.015])


def test_add_cross_pair_features_fills_neutral_zero_for_missing_timestamp(spark, synthetic_candles):
    df = spark.createDataFrame(synthetic_candles)
    ts0 = int(synthetic_candles["unix_epoch_s"].iloc[0])
    ts1 = int(synthetic_candles["unix_epoch_s"].iloc[1])
    # cross-pair data only covers the first timestamp -- second is a gap
    cross_pair_df = spark.createDataFrame(pd.DataFrame({
        "unix_epoch_s": [ts0], "usd_strength_return": [0.42],
    }))

    result = add_cross_pair_features(df, cross_pair_df).toPandas().set_index("unix_epoch_s")

    assert result.loc[ts0, "usd_strength_return"] == pytest.approx(0.42)
    assert result.loc[ts1, "usd_strength_return"] == 0.0


def test_add_targets_pd_lead_matches_manual_calculation(spark, synthetic_candles):
    lookahead = 4
    df = spark.createDataFrame(synthetic_candles)
    df = add_targets(df, lookahead)
    pdf = df.orderBy("unix_epoch_s").toPandas()

    mid_close = synthetic_candles["mid_close"]
    manual_pd_lead = 100.0 * (mid_close.shift(-lookahead) - mid_close) / mid_close

    np.testing.assert_allclose(
        pdf["pd_lead"].to_numpy()[:-lookahead],
        manual_pd_lead.to_numpy()[:-lookahead],
        rtol=1e-4,
    )
    # the last `lookahead` rows have no future bar to look ahead to
    assert pdf["pd_lead"].to_numpy()[-1:].size and np.isnan(pdf["pd_lead"].to_numpy()[-1])


def test_engineer_features_produces_full_n_back_windows(spark, synthetic_candles):
    n_back = 10
    ma_columns_list = ["volatility", "return", "diff_spread_close", "diff_volume"]

    df = spark.createDataFrame(synthetic_candles)
    df_time_series, df_non_time_series, columns_x = engineer_features(
        df,
        ma_lookback_list=[3, 5],
        ma_columns_list=ma_columns_list,
        columns_base=["mid_open", "mid_high", "mid_low", "mid_close", "spread_close", "volume"],
        lookahead=4,
        n_back=n_back,
        training_and_testing=True,
    )

    pdf = df_time_series.toPandas()
    assert len(pdf) > 0
    for col in columns_x:
        assert all(len(row) == n_back for row in pdf[col])

    # most raw OHLCV columns must be gone; engineered columns must remain
    for col in ["mid_open", "mid_high", "mid_low", "volume"]:
        assert col not in df_non_time_series.columns
    assert "pd_lead" in df_non_time_series.columns

    # mid_close/spread_close/realized_volatility are the exception (see
    # COLUMNS_PASSTHROUGH): kept as reference data for backtesting/position sizing,
    # but never fed to the model as a feature.
    assert "mid_close" in df_non_time_series.columns
    assert "spread_close" in df_non_time_series.columns
    assert "realized_volatility" in df_non_time_series.columns
    assert "mid_close" not in columns_x
    assert "spread_close" not in columns_x
    assert "realized_volatility" not in columns_x


def test_window_into_arrays_preserves_chronological_order_oldest_first(spark, synthetic_candles):
    """Pins down direction, not just shape/position: the windowed array for a given
    row must be [oldest, ..., current] — current bar last — because that's the order
    Keras LSTM (input_shape=(timesteps, features)) is built to consume, stepping
    through axis 1 from index 0 forward. A reversed window (current bar first) would
    still be shape-compatible and would still pass a position-only check with
    monotonic-looking data if that data happened to be symmetric — it would silently
    feed the model the sequence backwards in time. This test uses each row's own
    unix_epoch_s as the windowed value, so "oldest first, current last" is checked
    against ground truth (real timestamps), not merely internal consistency.
    """
    n_back = 5
    df = spark.createDataFrame(synthetic_candles).withColumn("marker", F.col("unix_epoch_s").cast("double"))

    windowed = window_into_arrays(df, ["marker"], n_back)
    pdf = windowed.orderBy("unix_epoch_s").toPandas()

    row = pdf.iloc[-1]  # the most recent row in the series
    window = [int(v) for v in row["marker"]]

    assert len(window) == n_back
    assert window == sorted(window)  # strictly ascending: oldest -> most recent
    assert window[-1] == row["unix_epoch_s"]  # current bar's own timestamp is last

    expected = [int(t) for t in synthetic_candles["unix_epoch_s"].to_numpy()[-n_back:]]
    assert window == expected


def test_session_features_fire_at_the_right_utc_hours(spark):
    """24 hourly bars starting at a known UTC midnight -- covers every hour exactly
    once, so each session flag can be checked against the literal expected pattern
    rather than just "some rows are True"."""
    start = int(datetime.datetime(2024, 1, 2, 0, 0, tzinfo=datetime.timezone.utc).timestamp())
    hours = list(range(24))
    pdf = pd.DataFrame({
        "instrument": "EUR/USD",
        "granularity": "H1",
        "unix_epoch_s": [start + h * 3600 for h in hours],
    })

    df = spark.createDataFrame(pdf)
    result = add_session_features(df).orderBy("unix_epoch_s").toPandas()

    assert list(result["is_tokyo_session"]) == [1.0 if 0 <= h < 9 else 0.0 for h in hours]
    assert list(result["is_london_session"]) == [1.0 if 8 <= h < 17 else 0.0 for h in hours]
    assert list(result["is_new_york_session"]) == [1.0 if 13 <= h < 22 else 0.0 for h in hours]
    assert list(result["is_london_new_york_overlap"]) == [1.0 if 13 <= h < 17 else 0.0 for h in hours]


def test_session_features_are_utc_independent_of_spark_session_timezone(spark):
    """Same 24-hour check, but with Spark's SQL session timezone deliberately set to
    something far from UTC -- proves the arithmetic UTC-hour computation isn't
    silently using F.hour(F.from_unixtime(...))'s timezone-dependent interpretation."""
    original_tz = spark.conf.get("spark.sql.session.timeZone")
    try:
        spark.conf.set("spark.sql.session.timeZone", "America/Los_Angeles")

        start = int(datetime.datetime(2024, 1, 2, 0, 0, tzinfo=datetime.timezone.utc).timestamp())
        hours = list(range(24))
        pdf = pd.DataFrame({
            "instrument": "EUR/USD",
            "granularity": "H1",
            "unix_epoch_s": [start + h * 3600 for h in hours],
        })

        df = spark.createDataFrame(pdf)
        result = add_session_features(df).orderBy("unix_epoch_s").toPandas()

        assert list(result["is_tokyo_session"]) == [1.0 if 0 <= h < 9 else 0.0 for h in hours]
    finally:
        spark.conf.set("spark.sql.session.timeZone", original_tz)
