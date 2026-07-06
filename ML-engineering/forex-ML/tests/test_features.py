from __future__ import annotations

import numpy as np
import pandas as pd

from forex_ml.data.features import (
    add_calendar_features,
    add_market_features,
    add_targets,
    engineer_features,
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

    # the raw OHLCV columns must be gone; engineered columns must remain
    for col in ["mid_open", "mid_high", "mid_low", "mid_close", "spread_close", "volume"]:
        assert col not in df_non_time_series.columns
    assert "pd_lead" in df_non_time_series.columns
