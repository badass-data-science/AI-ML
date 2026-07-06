"""Feature engineering for one (instrument, granularity) pair.

Ported from prepare-training-and-inference-data.ipynb, split into composable,
independently testable Spark transforms instead of one long notebook cell chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, Window
from pyspark.sql.types import FloatType
from python_tools_and_shortcuts.time_series_essentials.day_and_week.day_and_week import (
    day_cos, day_sin, week_cos, week_sin,
)

COLUMN_TIMESTAMP = "unix_epoch_s"
COLUMNS_PARTITION = ["instrument", "granularity"]
COLUMNS_Y = ["pd_lead", "spread_close_lead", "volatility_lead"]


@dataclass(frozen=True)
class FeatureColumns:
    """Names resolved once so every stage agrees on the sort/partition/target columns."""

    columns_partition: list[str] = field(default_factory=lambda: list(COLUMNS_PARTITION))
    column_timestamp: str = COLUMN_TIMESTAMP
    columns_y: list[str] = field(default_factory=lambda: list(COLUMNS_Y))

    @property
    def columns_sort(self) -> list[str]:
        return [*self.columns_partition, self.column_timestamp]


_day_sin_udf = F.udf(lambda ts: float(day_sin(ts)), FloatType())
_day_cos_udf = F.udf(lambda ts: float(day_cos(ts)), FloatType())
_week_sin_udf = F.udf(lambda ts: float(week_sin(ts)), FloatType())
_week_cos_udf = F.udf(lambda ts: float(week_cos(ts)), FloatType())


def add_calendar_features(df: DataFrame, cols: FeatureColumns = FeatureColumns()) -> DataFrame:
    ts = F.col(cols.column_timestamp)
    return (
        df
        .withColumn("day_sin", _day_sin_udf(ts))
        .withColumn("day_cos", _day_cos_udf(ts))
        .withColumn("week_sin", _week_sin_udf(ts))
        .withColumn("week_cos", _week_cos_udf(ts))
    )


def add_market_features(df: DataFrame, cols: FeatureColumns = FeatureColumns()) -> DataFrame:
    window_spec = Window.partitionBy(*cols.columns_partition).orderBy(*cols.columns_sort)
    return (
        df
        .withColumn("volatility", F.col("mid_high") - F.col("mid_low"))
        .withColumn("return", F.col("mid_close") - F.col("mid_open"))
        .withColumn("diff_spread_close", F.col("spread_close") - F.lag(F.col("spread_close"), 1).over(window_spec))
        .withColumn("diff_volume", F.col("volume") - F.lag(F.col("volume"), 1).over(window_spec))
    )


def add_targets(df: DataFrame, lookahead: int, cols: FeatureColumns = FeatureColumns()) -> DataFrame:
    window_spec = Window.partitionBy(*cols.columns_partition).orderBy(*cols.columns_sort)

    df = (
        df
        .withColumn("mid_close_lead", F.lead(F.col("mid_close"), lookahead).over(window_spec))
        .withColumn("pd_lead", 100.0 * (F.col("mid_close_lead") - F.col("mid_close")) / F.col("mid_close"))
        .drop("mid_close_lead")
        .withColumn("spread_close_lead", F.lead(F.col("spread_close"), lookahead).over(window_spec))
    )

    lookahead_window = window_spec.rowsBetween(1, lookahead)
    df = (
        df
        .withColumn("mid_high_lookahead", F.collect_list(F.col("mid_high")).over(lookahead_window))
        .withColumn("mid_low_lookahead", F.collect_list(F.col("mid_low")).over(lookahead_window))
        .withColumn("volatility_lead", F.array_max("mid_high_lookahead") - F.array_min("mid_low_lookahead"))
        .drop("mid_high_lookahead", "mid_low_lookahead")
    )
    return df


def compute_moving_averages(
    df: DataFrame,
    ma_lookback_list: list[int],
    ma_columns_list: list[str],
    cols: FeatureColumns = FeatureColumns(),
) -> DataFrame:
    for ma_lookback in ma_lookback_list:
        window_spec = (
            Window.partitionBy(*cols.columns_partition).orderBy(*cols.columns_sort).rowsBetween(1 - ma_lookback, 0)
        )
        for column_name in ma_columns_list:
            df = df.withColumn(f"{column_name}_MA_{ma_lookback}", F.avg(F.col(column_name)).over(window_spec))
    return df.orderBy(*cols.columns_sort)


def drop_raw_price_columns(
    df: DataFrame,
    columns_base: list[str],
    training_and_testing: bool,
    cols: FeatureColumns = FeatureColumns(),
) -> DataFrame:
    df = df.drop(*columns_base)
    if training_and_testing:
        df = df.dropna()
    return df.orderBy(*cols.columns_sort)


def add_row_number(df: DataFrame, cols: FeatureColumns = FeatureColumns()) -> DataFrame:
    window_spec = Window.partitionBy(*cols.columns_partition).orderBy(*cols.columns_sort)
    return df.withColumn("row_num", F.row_number().over(window_spec))


def filter_incomplete_rows(
    df: DataFrame,
    ma_lookback_list: list[int],
    lookahead: int,
    training_and_testing: bool,
    cols: FeatureColumns = FeatureColumns(),
) -> DataFrame:
    """Drop rows without a full moving-average history, and (when training) rows too
    close to the end of the series to have a complete lookahead target."""
    df = df.where(F.col("row_num") >= max(ma_lookback_list))
    if training_and_testing:
        count = df.count()
        df = df.where(F.col("row_num") <= count - lookahead)
    return df.orderBy(*cols.columns_sort).drop("row_num")


def select_xy_columns(df: DataFrame, cols: FeatureColumns = FeatureColumns()) -> tuple[DataFrame, list[str]]:
    to_select_sans_x = [*cols.columns_partition, cols.column_timestamp, *cols.columns_y]
    columns_x = [c for c in df.columns if c not in to_select_sans_x]
    to_select = [*to_select_sans_x, *columns_x]
    df = df.orderBy(*cols.columns_sort).select(*to_select)
    return df, columns_x


def window_into_arrays(
    df: DataFrame,
    columns_x: list[str],
    n_back: int,
    cols: FeatureColumns = FeatureColumns(),
) -> DataFrame:
    """Collect the trailing n_back bars for each feature into a per-row array — this is
    what turns flat rows into LSTM-shaped (n_back, n_features) windows."""
    window_spec = (
        Window.partitionBy(*cols.columns_partition).orderBy(*cols.columns_sort).rowsBetween(1 - n_back, 0)
    )

    df_lists = df.repartition(*cols.columns_partition)
    for column_name in columns_x:
        df_lists = df_lists.withColumn(column_name, F.collect_list(F.col(column_name)).over(window_spec))

    df_lists = df_lists.repartition(*cols.columns_partition).cache()
    df_lists = (
        df_lists
        .where(F.array_size(F.col(columns_x[0])) == F.lit(n_back))
        .orderBy(*cols.columns_sort)
    )
    return df_lists


def engineer_features(
    df: DataFrame,
    *,
    ma_lookback_list: list[int],
    ma_columns_list: list[str],
    columns_base: list[str],
    lookahead: int,
    n_back: int,
    training_and_testing: bool,
    cols: FeatureColumns = FeatureColumns(),
) -> tuple[DataFrame, DataFrame, list[str]]:
    """Run the full Stage-1 pipeline on raw candles.

    Returns (df_time_series_lists, df_non_time_series, columns_x) — the same two
    dataframe shapes the original notebook wrote to Parquet, plus the resolved
    feature-column list so callers don't have to re-derive it.
    """
    df = add_calendar_features(df, cols)
    df = add_market_features(df, cols)
    df = add_targets(df, lookahead, cols)
    df = drop_raw_price_columns(df, columns_base, training_and_testing, cols)
    df = add_row_number(df, cols)
    df = compute_moving_averages(df, ma_lookback_list, ma_columns_list, cols)
    df = filter_incomplete_rows(df, ma_lookback_list, lookahead, training_and_testing, cols)

    df_non_time_series, columns_x = select_xy_columns(df, cols)
    df_time_series = window_into_arrays(df_non_time_series, columns_x, n_back, cols)

    return df_time_series, df_non_time_series, columns_x
