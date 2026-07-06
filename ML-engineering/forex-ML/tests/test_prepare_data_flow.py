from __future__ import annotations

from forex_ml.config import FeatureParams
from forex_ml.flows.prepare_data_flow import engineer_and_save_task

FEATURE_PARAMS = FeatureParams(
    instruments=["EUR/USD"],
    granularities=["H1"],
    n_back=10,
    lookahead=2,
    ma_lookback_list=[3, 5],
    columns_base=["mid_open", "mid_high", "mid_low", "mid_close", "spread_close", "volume"],
    ma_columns_list=["volatility", "return", "diff_spread_close", "diff_volume"],
    training_and_testing=True,
    min_training_timestamp="2020-01-01T00:00:00",
    output_dir="unused",  # overridden per-test via tmp_path
)


def test_is_forward_filled_survives_through_to_columns_x_when_present(spark, synthetic_candles, tmp_path):
    pdf = synthetic_candles.copy()
    pdf["is_forward_filled"] = [i % 7 == 0 for i in range(len(pdf))]  # some True, some False

    params = FEATURE_PARAMS.model_copy(update={"output_dir": str(tmp_path)})
    key = engineer_and_save_task(spark, pdf, "EUR/USD", "H1", params)

    from forex_ml.paths import time_series_parquet_path
    result = spark.read.parquet(str(time_series_parquet_path(tmp_path, key)))
    assert "is_forward_filled" in result.columns


def test_missing_is_forward_filled_column_does_not_break_the_pipeline(spark, synthetic_candles, tmp_path):
    pdf = synthetic_candles.copy()  # no is_forward_filled column at all
    assert "is_forward_filled" not in pdf.columns

    params = FEATURE_PARAMS.model_copy(update={"output_dir": str(tmp_path)})
    key = engineer_and_save_task(spark, pdf, "EUR/USD", "H1", params)

    from forex_ml.paths import time_series_parquet_path
    result = spark.read.parquet(str(time_series_parquet_path(tmp_path, key)))
    assert "is_forward_filled" not in result.columns
    assert result.count() > 0
