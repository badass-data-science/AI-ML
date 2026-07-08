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

# Reference-only columns -- never fed to the model as a feature (see
# select_xy_columns/window_into_arrays, which only window columns NOT in this list),
# but needed downstream. mid_close/spread_close: forex-strategy's backtest needs to
# know the actual price a test-set row traded at and the spread cost of trading it;
# everything else in columns_base (mid_open/high/low, volume) is still dropped.
# realized_volatility (see add_market_features): a fixed-window, backward-looking
# realized-volatility reference for position sizing -- forex-strategy scales trade
# size down as this rises, rather than needing a second, forward-looking volatility
# model. Deliberately NOT one of the ma_lookback_list-configurable volatility_MA_N
# columns: those depend on a config value that varies across environments (e.g.
# every test here uses ma_lookback_list=[3, 5], production uses [12, 30, 50]) --
# this needs to exist unconditionally, the same reason mid_close/spread_close do.
COLUMNS_PASSTHROUGH = ["mid_close", "spread_close", "realized_volatility"]

# Approximate UTC session windows (standard interbank FX convention: half-open
# [start, end) hours). Fixed UTC hours, not DST-aware — London/New York local session
# times shift by an hour twice a year, which this simplification ignores. Good enough
# as an indicator feature; not precise enough for exact session-open/close timing.
TOKYO_SESSION_UTC = (0, 9)
LONDON_SESSION_UTC = (8, 17)
NEW_YORK_SESSION_UTC = (13, 22)


@dataclass(frozen=True)
class FeatureColumns:
    """Names resolved once so every stage agrees on the sort/partition/target columns."""

    columns_partition: list[str] = field(default_factory=lambda: list(COLUMNS_PARTITION))
    column_timestamp: str = COLUMN_TIMESTAMP
    columns_y: list[str] = field(default_factory=lambda: list(COLUMNS_Y))
    columns_passthrough: list[str] = field(default_factory=lambda: list(COLUMNS_PASSTHROUGH))

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


def add_session_features(df: DataFrame, cols: FeatureColumns = FeatureColumns()) -> DataFrame:
    """Trading-session indicators (Tokyo/London/New York + the London-NY overlap,
    historically the most liquid/volatile window) — a well-documented FX volatility
    driver distinct from the existing day/week cyclical features, which capture
    broad calendar seasonality but not which major market is currently open.
    """
    # UTC hour via plain arithmetic on the epoch second — NOT F.hour(F.from_unixtime(
    # ...)), which interprets the timestamp using Spark's SQL session timezone (JVM
    # default unless configured) and would silently shift every session boundary if
    # that timezone isn't UTC. Unix epoch seconds are UTC by definition, so this is
    # the only way to get the UTC hour independent of any session/JVM timezone
    # setting — the same principle day_sin/week_sin above already rely on.
    utc_hour = (F.col(cols.column_timestamp) % 86400) / 3600

    tokyo_lo, tokyo_hi = TOKYO_SESSION_UTC
    london_lo, london_hi = LONDON_SESSION_UTC
    ny_lo, ny_hi = NEW_YORK_SESSION_UTC

    return (
        df
        .withColumn("is_tokyo_session", ((utc_hour >= tokyo_lo) & (utc_hour < tokyo_hi)).cast("double"))
        .withColumn("is_london_session", ((utc_hour >= london_lo) & (utc_hour < london_hi)).cast("double"))
        .withColumn("is_new_york_session", ((utc_hour >= ny_lo) & (utc_hour < ny_hi)).cast("double"))
        # [ny_lo, london_hi) is the intersection of LONDON_SESSION_UTC and
        # NEW_YORK_SESSION_UTC given the specific constants above (NY starts later
        # than London, London ends earlier than NY) — not a general intersection
        # formula. Re-derive this if those constants ever change.
        .withColumn(
            "is_london_new_york_overlap", ((utc_hour >= ny_lo) & (utc_hour < london_hi)).cast("double"),
        )
    )


_REALIZED_VOLATILITY_WINDOW_BARS = 12


def add_market_features(df: DataFrame, cols: FeatureColumns = FeatureColumns()) -> DataFrame:
    window_spec = Window.partitionBy(*cols.columns_partition).orderBy(*cols.columns_sort)
    df = (
        df
        .withColumn("volatility", F.col("mid_high") - F.col("mid_low"))
        .withColumn("return", F.col("mid_close") - F.col("mid_open"))
        .withColumn("diff_spread_close", F.col("spread_close") - F.lag(F.col("spread_close"), 1).over(window_spec))
        .withColumn("diff_volume", F.col("volume") - F.lag(F.col("volume"), 1).over(window_spec))
    )
    # A fixed-window, backward-looking realized-volatility reference (see
    # COLUMNS_PASSTHROUGH) -- deliberately NOT one of the ma_lookback_list-configurable
    # volatility_MA_N columns below, so it exists unconditionally regardless of
    # whatever moving-average windows happen to be configured for actual model
    # features. Used by forex-strategy for position sizing, not as a model input.
    realized_vol_window = window_spec.rowsBetween(1 - _REALIZED_VOLATILITY_WINDOW_BARS, 0)
    return df.withColumn("realized_volatility", F.avg(F.col("volatility")).over(realized_vol_window))


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
    """Drops columns_base EXCEPT cols.columns_passthrough -- those survive as
    reference data (see COLUMNS_PASSTHROUGH) rather than being dropped like the rest
    of the raw OHLCV columns."""
    to_drop = [c for c in columns_base if c not in cols.columns_passthrough]
    df = df.drop(*to_drop)
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
    """columns_passthrough is excluded from columns_x here (not a model input) but
    still included in `to_select`, so it rides along as a flat, unwindowed reference
    column through window_into_arrays (which only windows columns named in columns_x,
    leaving everything else in the row untouched)."""
    to_select_sans_x = [*cols.columns_partition, cols.column_timestamp, *cols.columns_y, *cols.columns_passthrough]
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
    df = add_session_features(df, cols)
    df = add_market_features(df, cols)
    df = add_targets(df, lookahead, cols)
    df = drop_raw_price_columns(df, columns_base, training_and_testing, cols)
    df = add_row_number(df, cols)
    df = compute_moving_averages(df, ma_lookback_list, ma_columns_list, cols)
    df = filter_incomplete_rows(df, ma_lookback_list, lookahead, training_and_testing, cols)

    df_non_time_series, columns_x = select_xy_columns(df, cols)
    df_time_series = window_into_arrays(df_non_time_series, columns_x, n_back, cols)

    return df_time_series, df_non_time_series, columns_x
