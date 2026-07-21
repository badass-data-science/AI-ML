"""Feature engineering for one (instrument, granularity) pair.

Ported from prepare-training-and-inference-data.ipynb, split into composable,
independently testable Spark transforms instead of one long notebook cell chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import DataFrame, Window
from pyspark.sql.types import DoubleType, FloatType
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


# Pairs where USD is the BASE currency (USD/XXX) -- a rising price means USD
# strengthened, so their return contributes to compute_cross_pair_usd_strength
# directly (+). Every other pair this project tracks is USD-QUOTE (XXX/USD),
# where a rising price means the OTHER currency strengthened (USD weakened), so
# those contribute with a flipped sign (-). See compute_cross_pair_usd_strength.
USD_BASE_PAIRS = frozenset({"USD/CAD", "USD/CHF", "USD/JPY"})


def compute_cross_pair_usd_strength(
    other_pairs_returns: dict[str, DataFrame],
    cols: FeatureColumns = FeatureColumns(),
) -> DataFrame:
    """Averages sign-adjusted returns across this project's OTHER configured pairs
    into one (unix_epoch_s, usd_strength_return) frame -- a broad-dollar-strength
    proxy, since every pair this project tracks is USD-quoted or USD-based. Each
    value in `other_pairs_returns` must already be reduced to just
    [unix_epoch_s, return] for that instrument (the caller's job -- see
    prepare_data_flow.py's pull_cross_pair_return_task, which computes this
    directly from that pair's raw candles rather than running its full Stage-1
    pipeline, since only the raw return is needed here). See USD_BASE_PAIRS for
    the sign convention.
    """
    parts = []
    for instrument, other_df in other_pairs_returns.items():
        sign = 1.0 if instrument in USD_BASE_PAIRS else -1.0
        parts.append(
            other_df.select(
                F.col(cols.column_timestamp),
                (F.lit(sign) * F.col("return")).alias("signed_return"),
            )
        )
    stacked = parts[0]
    for part in parts[1:]:
        stacked = stacked.unionByName(part)
    return stacked.groupBy(cols.column_timestamp).agg(F.avg("signed_return").alias("usd_strength_return"))


def add_cross_pair_features(
    df: DataFrame,
    cross_pair_usd_strength: DataFrame,
    cols: FeatureColumns = FeatureColumns(),
) -> DataFrame:
    """Joins a precomputed cross-pair "USD strength" signal (see
    compute_cross_pair_usd_strength) onto this pair's frame by timestamp. Left
    join: a timestamp with no cross-pair data (e.g. every other pair's feed has a
    gap right there) gets a neutral 0.0 rather than dropping the row -- degrade
    gracefully, same principle as is_forward_filled elsewhere in this pipeline.
    """
    joined = df.join(cross_pair_usd_strength, on=cols.column_timestamp, how="left")
    return joined.withColumn("usd_strength_return", F.coalesce(F.col("usd_strength_return"), F.lit(0.0)))


_SECONDS_PER_DAY = 24 * 60 * 60


def add_daily_timeframe_features(
    df: DataFrame,
    daily_trend: pd.DataFrame,
    cols: FeatureColumns = FeatureColumns(),
) -> DataFrame:
    """As-of join: attaches the most recently CLOSED daily bar's own trailing
    return/volatility average onto every row of `df` (H1 or any other sub-daily
    granularity), via a broadcast binary-search lookup rather than an
    exact-timestamp match (a daily bar and an H1 bar essentially never share a
    timestamp).

    `daily_trend` is a small, already-computed pandas DataFrame (see
    prepare_data_flow.py's pull_daily_trend_task) with one row per daily bar:
    [unix_epoch_s, daily_return_ma, daily_volatility_ma].

    Causal correctness: a daily bar spanning [t, t+86400) does not actually CLOSE
    until t+86400, so using its own start-of-day unix_epoch_s as the "available
    from" cutoff would let every H1 bar on that SAME calendar day see that day's
    own still-forming aggregate -- a lookahead bug of exactly the kind this
    project has hunted before (see ForwardFillInator's DST fix,
    triple_barrier.py's purge_bars). Shifting each daily bar's timestamp forward
    by one full day before the lookup fixes this: an H1 bar can only ever see a
    daily bar that has genuinely finished.
    """
    daily_sorted = daily_trend.sort_values(cols.column_timestamp).reset_index(drop=True)
    available_from = daily_sorted[cols.column_timestamp].to_numpy() + _SECONDS_PER_DAY

    def _make_lookup_udf(values: np.ndarray):
        @F.pandas_udf(DoubleType())  # type: ignore[call-overload]  # pyspark stub gap, not a real type error
        def _lookup(timestamp: pd.Series) -> pd.Series:
            idx = np.searchsorted(available_from, timestamp.to_numpy(), side="right") - 1
            result = np.where(idx >= 0, values[np.clip(idx, 0, len(values) - 1)], np.nan)
            return pd.Series(result, index=timestamp.index)

        return _lookup

    return (
        df
        .withColumn("daily_return_ma", _make_lookup_udf(daily_sorted["daily_return_ma"].to_numpy())(F.col(cols.column_timestamp)))
        .withColumn("daily_volatility_ma", _make_lookup_udf(daily_sorted["daily_volatility_ma"].to_numpy())(F.col(cols.column_timestamp)))
    )


# Sub-window sizes for the rescaled-range Hurst estimator below -- must each divide
# HURST_WINDOW_BARS evenly (100/10=10, 100/20=5, 100/25=4, 100/50=2, 100/100=1 chunks)
# so every scale gets at least one full, non-overlapping chunk. Five log-spaced
# scales is enough points for a stable log-log slope without over-fragmenting a
# 100-bar window into chunks too short to estimate a meaningful local R/S.
HURST_WINDOW_BARS = 100
HURST_SUBWINDOW_SIZES = (10, 20, 25, 50, 100)


def _rescaled_range_hurst(
    returns: np.ndarray,
    subwindow_sizes: tuple[int, ...] = HURST_SUBWINDOW_SIZES,
) -> float:
    """Classic rescaled-range (R/S) Hurst exponent estimate (Mandelbrot & Wallis
    1969), applied directly to a window of (already-stationary) per-bar `return`
    values -- the standard financial/econophysics convention (e.g. Peters,
    *Fractal Market Analysis*), rather than to raw non-stationary price levels.
    H > 0.5 indicates persistent/trending behavior, H < 0.5 mean-reverting, H = 0.5
    an uncorrelated (efficient-market-like) return series.

    Splits `returns` into non-overlapping chunks at each size in `subwindow_sizes`,
    averages R/S across same-size chunks, then fits the slope of log(mean R/S) vs
    log(size) across scales -- more robust than reading R/S off a single window
    size. Returns NaN if there's not enough data to fit at least two distinct
    scales (can't determine a slope from fewer than two points).
    """
    n = len(returns)
    log_sizes: list[float] = []
    log_rs: list[float] = []
    for size in subwindow_sizes:
        if size > n or size < 2:
            continue
        n_chunks = n // size
        rs_values = []
        for i in range(n_chunks):
            chunk = returns[i * size:(i + 1) * size]
            deviations = np.cumsum(chunk - chunk.mean())
            r = deviations.max() - deviations.min()
            s = chunk.std()
            if s > 0:
                rs_values.append(r / s)
        if rs_values:
            log_sizes.append(np.log(size))
            log_rs.append(np.log(np.mean(rs_values)))
    if len(log_sizes) < 2:
        return float("nan")
    slope, _ = np.polyfit(log_sizes, log_rs, 1)
    return float(slope)


def add_rolling_hurst_feature(
    df: DataFrame,
    window_bars: int = HURST_WINDOW_BARS,
    cols: FeatureColumns = FeatureColumns(),
) -> DataFrame:
    """Rolling Hurst exponent over a trailing window of `return` -- see
    _rescaled_range_hurst. Causal by construction: the trailing window
    (`rowsBetween(1-window_bars, 0)`) never includes a future bar, the same
    convention every other rolling feature in this module already uses (e.g.
    compute_moving_averages)."""
    window_spec = (
        Window.partitionBy(*cols.columns_partition).orderBy(*cols.columns_sort)
        .rowsBetween(1 - window_bars, 0)
    )
    df = df.withColumn("_hurst_window", F.collect_list(F.col("return")).over(window_spec))

    @F.pandas_udf(DoubleType())  # type: ignore[call-overload]  # pyspark stub gap, not a real type error
    def _hurst(windows: pd.Series) -> pd.Series:
        return pd.Series([_rescaled_range_hurst(np.asarray(w)) for w in windows], index=windows.index)

    return df.withColumn("hurst_exponent", _hurst(F.col("_hurst_window"))).drop("_hurst_window")


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


def add_volatility_regime_features(
    df: DataFrame,
    ma_lookback_list: list[int],
    cols: FeatureColumns = FeatureColumns(),
) -> DataFrame:
    """Ratio of short- to long-horizon realized volatility (volatility_MA_<min
    lookback> / volatility_MA_<max lookback>) -- expanding (>1) vs. contracting (<1)
    volatility regime, a signal distinct from either MA's own absolute level. Must
    run after compute_moving_averages with "volatility" in ma_columns_list (needs
    both MA columns to already exist) -- derives which two lookbacks to use from
    ma_lookback_list itself, rather than hardcoding specific values, so this doesn't
    break under a different-than-production lookback list (e.g. a scaled-down test
    config). Guards near-zero long-window volatility (a long enough flat stretch)
    the same way persistence_baseline guards its own zero-denominator edge case:
    fall back to a neutral ratio of 1.0 rather than dividing by ~0.
    """
    short_col = F.col(f"volatility_MA_{min(ma_lookback_list)}")
    long_col = F.col(f"volatility_MA_{max(ma_lookback_list)}")
    ratio = F.when(long_col > 1e-9, short_col / long_col).otherwise(F.lit(1.0))
    return df.withColumn("volatility_regime_ratio", ratio)


def add_momentum_features(
    df: DataFrame,
    ma_lookback_list: list[int],
    cols: FeatureColumns = FeatureColumns(),
) -> DataFrame:
    """Stationarity-safe analogs of classic price-momentum oscillators, built on
    the already-stationary `return` column rather than raw price levels (this
    project's existing convention -- see drop_raw_price_columns) -- not the
    textbook price-based MACD/RSI/Bollinger-band formulas, which Spark window
    functions can't express directly anyway without a per-partition UDF (Wilder's
    RSI smoothing and MACD's EMA are both recursive across rows). Reuses
    ma_lookback_list's own min/max, the same way add_volatility_regime_features
    does, rather than introducing new hardcoded lookback constants:

    - `return_sma_crossover` = return_MA_<min> - return_MA_<max> (an SMA-crossover
      MACD analog -- positive when short-horizon trend runs above long-horizon
      trend). Needs compute_moving_averages to have already run.
    - `return_zscore_<min>` = (return - return_MA_<min>) / stddev(return) over the
      same trailing <min>-bar window (a Bollinger-band-position analog). Guards a
      near-zero trailing stddev (a long flat stretch) by falling back to 0.
    - `rsi_<min>` = a simple (non-Wilder-smoothed) RSI variant over the trailing
      <min>-bar window: avg(positive returns) / [avg(positive) + avg(negative
      magnitude)], scaled to the usual 0-100 range. Guards the case where both
      trailing gain and loss are ~0 (a flat stretch) by falling back to a neutral
      50 rather than dividing by ~0.
    """
    short_lookback = min(ma_lookback_list)
    long_lookback = max(ma_lookback_list)

    df = df.withColumn(
        "return_sma_crossover",
        F.col(f"return_MA_{short_lookback}") - F.col(f"return_MA_{long_lookback}"),
    )

    window_spec = (
        Window.partitionBy(*cols.columns_partition).orderBy(*cols.columns_sort)
        .rowsBetween(1 - short_lookback, 0)
    )
    trailing_std = F.stddev(F.col("return")).over(window_spec)
    zscore = F.when(
        trailing_std > 1e-12, (F.col("return") - F.col(f"return_MA_{short_lookback}")) / trailing_std,
    ).otherwise(F.lit(0.0))
    df = df.withColumn(f"return_zscore_{short_lookback}", zscore)

    gain = F.when(F.col("return") > 0, F.col("return")).otherwise(F.lit(0.0))
    loss = F.when(F.col("return") < 0, -F.col("return")).otherwise(F.lit(0.0))
    avg_gain = F.avg(gain).over(window_spec)
    avg_loss = F.avg(loss).over(window_spec)
    rsi = F.when(
        (avg_gain + avg_loss) > 1e-12, 100.0 * avg_gain / (avg_gain + avg_loss),
    ).otherwise(F.lit(50.0))
    return df.withColumn(f"rsi_{short_lookback}", rsi)


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
    min_history_bars: int = 0,
) -> DataFrame:
    """Drop rows without a full moving-average (or other rolling-feature) history,
    and (when training) rows too close to the end of the series to have a complete
    lookahead target. `min_history_bars` lets a caller with a rolling window
    longer than every ma_lookback_list entry (e.g. add_rolling_hurst_feature's
    HURST_WINDOW_BARS) fold that requirement in too, rather than only ever
    checking ma_lookback_list -- otherwise rows with a full MA history but an
    incomplete Hurst window would silently pass this filter with a NaN feature."""
    df = df.where(F.col("row_num") >= max(max(ma_lookback_list), min_history_bars))
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
    cross_pair_usd_strength: DataFrame | None = None,
    daily_trend: pd.DataFrame | None = None,
    include_hurst: bool = False,
    cols: FeatureColumns = FeatureColumns(),
) -> tuple[DataFrame, DataFrame, list[str]]:
    """Run the full Stage-1 pipeline on raw candles.

    Returns (df_time_series_lists, df_non_time_series, columns_x) — the same two
    dataframe shapes the original notebook wrote to Parquet, plus the resolved
    feature-column list so callers don't have to re-derive it.

    `cross_pair_usd_strength` is optional (default None, meaning skip) because it
    needs sibling-pair data the caller must assemble via I/O (see
    prepare_data_flow.py's pull_cross_pair_return_task/compute_cross_pair_usd_strength)
    -- this function itself stays pure/I/O-free. Production runs always pass it;
    tests that only care about a single synthetic pair can omit it.

    `daily_trend` is likewise optional (default None, meaning skip) -- this
    pair's OWN daily-bar trend/volatility, assembled by the caller via
    prepare_data_flow.py's pull_daily_trend_task, see add_daily_timeframe_features.

    `include_hurst` adds a rolling Hurst exponent (see add_rolling_hurst_feature) --
    no external data needed (pure function of this pair's own `return` column), so
    unlike the two flags above this is a plain bool, not an optional DataFrame/pdf.
    """
    df = add_calendar_features(df, cols)
    df = add_session_features(df, cols)
    df = add_market_features(df, cols)
    if cross_pair_usd_strength is not None:
        df = add_cross_pair_features(df, cross_pair_usd_strength, cols)
    if daily_trend is not None:
        df = add_daily_timeframe_features(df, daily_trend, cols)
    if include_hurst:
        df = add_rolling_hurst_feature(df, cols=cols)
    df = add_targets(df, lookahead, cols)
    df = drop_raw_price_columns(df, columns_base, training_and_testing, cols)
    df = add_row_number(df, cols)
    df = compute_moving_averages(df, ma_lookback_list, ma_columns_list, cols)
    df = add_volatility_regime_features(df, ma_lookback_list, cols)
    df = add_momentum_features(df, ma_lookback_list, cols)
    min_history_bars = HURST_WINDOW_BARS if include_hurst else 0
    df = filter_incomplete_rows(df, ma_lookback_list, lookahead, training_and_testing, cols, min_history_bars)

    df_non_time_series, columns_x = select_xy_columns(df, cols)
    df_time_series = window_into_arrays(df_non_time_series, columns_x, n_back, cols)

    return df_time_series, df_non_time_series, columns_x
