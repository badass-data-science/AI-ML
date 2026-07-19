"""Prefect flow: pull candles -> engineer features -> window -> Parquet, for one
(instrument, granularity) pair. Replaces prepare-training-and-inference-data.ipynb.

Run ad-hoc:
    python -m forex_ml.flows.prepare_data_flow --instrument EUR/USD --granularity H1

Spark memory sizing is a `--spark-memory` CLI option (default 70g) rather than
hardcoded -- see forex_ml.spark_session for why.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
import pyspark.sql.functions as F
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE
from pyspark.sql import DataFrame, SparkSession

from forex_ml.config import FeatureParams, load_params
from forex_ml.data import influx_source
from forex_ml.data.features import compute_cross_pair_usd_strength, engineer_features
from forex_ml.paths import non_time_series_parquet_path, pair_key, stage1_config_path, time_series_parquet_path
from forex_ml.spark_session import DEFAULT_SPARK_MEMORY, build_spark_session


def _pull_candles(instrument: str, granularity: str, params: FeatureParams):
    from forex.eda.eda_config.eda_config import granularity_to_seconds_map

    if params.training_and_testing:
        min_ts, max_ts = influx_source.training_timestamp_range(params.min_training_timestamp)
    else:
        min_ts, max_ts = influx_source.inference_timestamp_range(
            granularity_to_seconds_map[granularity], params.n_back
        )
    return influx_source.pull_candles(instrument, granularity, min_ts, max_ts)


@task(name="pull-candles", retries=3, retry_delay_seconds=30)
def pull_candles_task(instrument: str, granularity: str, params: FeatureParams):
    logger = get_run_logger()
    pdf = _pull_candles(instrument, granularity, params)
    logger.info("Pulled %d rows for %s %s", len(pdf), instrument, granularity)
    return pdf


@task(name="pull-cross-pair-return", retries=3, retry_delay_seconds=30)
def pull_cross_pair_return_task(instrument: str, granularity: str, params: FeatureParams):
    """Pulls one OTHER pair's raw candles (same source/date-range as the target
    pair) and reduces to just [unix_epoch_s, return] -- return = mid_close -
    mid_open, matching add_market_features's own definition for the target pair,
    computed directly here rather than running that other pair's full Stage-1
    pipeline, which compute_cross_pair_usd_strength doesn't need.
    """
    logger = get_run_logger()
    pdf = _pull_candles(instrument, granularity, params)
    logger.info("Pulled %d cross-pair rows for %s %s", len(pdf), instrument, granularity)
    reduced = pdf[["unix_epoch_s", "mid_open", "mid_close"]].copy()
    reduced["return"] = reduced["mid_close"] - reduced["mid_open"]
    return reduced[["unix_epoch_s", "return"]]


DAILY_TREND_MA_LOOKBACK_DAYS = 5


@task(name="pull-daily-trend", retries=3, retry_delay_seconds=30)
def pull_daily_trend_task(instrument: str, params: FeatureParams, ma_lookback_days: int = DAILY_TREND_MA_LOOKBACK_DAYS) -> pd.DataFrame:
    """Pulls this SAME pair's own Daily-granularity candles and reduces to a short
    trailing average of daily return/volatility -- see
    forex_ml.data.features.add_daily_timeframe_features for how this gets attached
    back onto the (typically much finer-granularity) target frame via a
    causally-correct as-of join. Requires Daily candles to already exist in
    InfluxDB for this instrument (forex-etl's candlestick_flow/forward_fill_flow
    at granularity='D') -- not every tracked pair has this backfilled yet.
    """
    logger = get_run_logger()
    pdf = _pull_candles(instrument, "D", params).sort_values("unix_epoch_s").reset_index(drop=True)
    daily_return = pdf["mid_close"] - pdf["mid_open"]
    daily_volatility = pdf["mid_high"] - pdf["mid_low"]
    result = pd.DataFrame({
        "unix_epoch_s": pdf["unix_epoch_s"],
        "daily_return_ma": daily_return.rolling(ma_lookback_days, min_periods=1).mean(),
        "daily_volatility_ma": daily_volatility.rolling(ma_lookback_days, min_periods=1).mean(),
    })
    logger.info("Pulled %d daily bars for %s daily-trend feature", len(result), instrument)
    return result


@task(name="engineer-and-save-features", cache_policy=NO_CACHE)
def engineer_and_save_task(
    spark: SparkSession,
    pdf,
    instrument: str,
    granularity: str,
    params: FeatureParams,
    other_pairs_returns: dict[str, DataFrame] | None = None,
    daily_trend: pd.DataFrame | None = None,
) -> str:
    # NO_CACHE: this task's args include a SparkSession/DataFrame, which Prefect's
    # default cache-key hashing can't serialize — without it, every run logs a noisy
    # (harmless) HashError and just skips caching anyway, so opt out explicitly.
    logger = get_run_logger()
    columns_sort = ["instrument", "granularity", "unix_epoch_s"]

    df: DataFrame = spark.createDataFrame(pdf)
    df = (
        df
        .select(*columns_sort, *[c for c in df.columns if c not in columns_sort])
        .orderBy(*columns_sort)
        .repartition("instrument", "granularity")
    )
    # is_forward_filled is populated by forex-etl's ForwardFillInator, but historical
    # data written before that field existed won't have it — carry it through as an
    # available (not mandatory) feature when present, rather than requiring it.
    optional_columns = []
    if "is_forward_filled" in df.columns:
        optional_columns.append(F.col("is_forward_filled").cast("double").alias("is_forward_filled"))

    # Insurance: the Flux query already filtered to this pair, but re-filter in case
    # the underlying measurement ever contains overlapping data.
    df = (
        df
        .where(F.col("instrument") == instrument)
        .where(F.col("granularity") == granularity)
        .select(*columns_sort, *params.columns_base, *optional_columns)
    )

    cross_pair_usd_strength = None
    if other_pairs_returns:
        cross_pair_usd_strength = compute_cross_pair_usd_strength(other_pairs_returns)

    df_time_series, df_non_time_series, columns_x = engineer_features(
        df,
        ma_lookback_list=params.ma_lookback_list,
        ma_columns_list=params.ma_columns_list,
        columns_base=params.columns_base,
        lookahead=params.lookahead,
        n_back=params.n_back,
        training_and_testing=params.training_and_testing,
        cross_pair_usd_strength=cross_pair_usd_strength,
        daily_trend=daily_trend,
    )

    key = pair_key(instrument, granularity, params.n_back, params.lookahead)
    ts_path = time_series_parquet_path(params.output_dir, key)
    non_ts_path = non_time_series_parquet_path(params.output_dir, key)
    config_path = stage1_config_path(params.output_dir, key)

    (
        df_time_series.repartition("instrument", "granularity").orderBy(*columns_sort)
        .write.mode("overwrite").parquet(str(ts_path))
    )
    (
        df_non_time_series.repartition("instrument", "granularity").orderBy(*columns_sort)
        .write.mode("overwrite").parquet(str(non_ts_path))
    )

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({
        "instrument": instrument,
        "granularity": granularity,
        "n_back": params.n_back,
        "lookahead": params.lookahead,
        "columns_x": columns_x,
    }, indent=2))

    logger.info("Wrote Stage-1 output for %s %s -> %s", instrument, granularity, key)
    return key


@flow(name="forex-ml-prepare-data", log_prints=True)
def prepare_data_flow(
    instrument: str, granularity: str, params_path: str | None = None, spark_memory: str = DEFAULT_SPARK_MEMORY,
) -> str:
    """Does NOT stop the SparkSession it gets/creates — a JVM only ever has one active
    SparkContext, so stopping it here would kill it out from under any other flow
    (prepare_all_flow, serve.py's retrain loop) or test fixture sharing the same
    process. Session lifecycle is the caller/process's responsibility; on a one-shot
    CLI run the JVM tears down naturally when the process exits."""
    params = load_params(params_path) if params_path else load_params()
    spark = build_spark_session("forex-ml-prepare-data", memory=spark_memory)
    pdf = pull_candles_task(instrument, granularity, params.feature)

    # Cross-pair "USD strength" feature (see compute_cross_pair_usd_strength) needs
    # every OTHER configured pair's raw return, pulled fresh here rather than reused
    # from that pair's own Stage-1 output -- avoids an ordering dependency where
    # pair A's prep would require pair B's Stage-1 to already exist (and vice versa).
    other_instruments = [i for i in params.feature.instruments if i != instrument]
    other_pairs_returns = {
        other_instrument: spark.createDataFrame(
            pull_cross_pair_return_task(other_instrument, granularity, params.feature)
        )
        for other_instrument in other_instruments
    }

    daily_trend = pull_daily_trend_task(instrument, params.feature) if params.feature.include_daily_trend else None

    return engineer_and_save_task(spark, pdf, instrument, granularity, params.feature, other_pairs_returns, daily_trend)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Stage-1 features for one (instrument, granularity) pair.")
    parser.add_argument("--instrument", required=True, help="e.g. EUR/USD")
    parser.add_argument("--granularity", required=True, help="e.g. H1")
    parser.add_argument("--params", default=None, help="Path to params.yaml (default: repo root)")
    parser.add_argument(
        "--spark-memory", default=DEFAULT_SPARK_MEMORY,
        help=f"spark.driver.memory / spark.executor.memory / spark.driver.maxResultSize (default: {DEFAULT_SPARK_MEMORY})",
    )
    args = parser.parse_args()
    prepare_data_flow(args.instrument, args.granularity, args.params, spark_memory=args.spark_memory)


if __name__ == "__main__":
    main()
