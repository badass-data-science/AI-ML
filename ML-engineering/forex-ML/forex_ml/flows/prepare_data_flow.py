"""Prefect flow: pull candles -> engineer features -> window -> Parquet, for one
(instrument, granularity) pair. Replaces prepare-training-and-inference-data.ipynb.

Run ad-hoc:
    python -m forex_ml.flows.prepare_data_flow --instrument EUR/USD --granularity H1

Spark memory/executor tuning is deliberately not hardcoded here (the original
notebooks hardcoded 70G/100G executor+driver memory, which only made sense on one
specific workstation) — configure it externally via spark-defaults.conf or
SPARK_* environment variables for your own hardware.
"""

from __future__ import annotations

import argparse
import json

import pyspark.sql.functions as F
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE
from pyspark.sql import DataFrame, SparkSession

from forex_ml.config import FeatureParams, load_params
from forex_ml.data import influx_source
from forex_ml.data.features import engineer_features
from forex_ml.paths import non_time_series_parquet_path, pair_key, stage1_config_path, time_series_parquet_path


@task(name="pull-candles", retries=3, retry_delay_seconds=30)
def pull_candles_task(instrument: str, granularity: str, params: FeatureParams):
    from forex.eda.eda_config.eda_config import granularity_to_seconds_map

    logger = get_run_logger()
    if params.training_and_testing:
        min_ts, max_ts = influx_source.training_timestamp_range(params.min_training_timestamp)
    else:
        min_ts, max_ts = influx_source.inference_timestamp_range(
            granularity_to_seconds_map[granularity], params.n_back
        )

    pdf = influx_source.pull_candles(instrument, granularity, min_ts, max_ts)
    logger.info("Pulled %d rows for %s %s", len(pdf), instrument, granularity)
    return pdf


@task(name="engineer-and-save-features", cache_policy=NO_CACHE)
def engineer_and_save_task(spark: SparkSession, pdf, instrument: str, granularity: str, params: FeatureParams) -> str:
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

    df_time_series, df_non_time_series, columns_x = engineer_features(
        df,
        ma_lookback_list=params.ma_lookback_list,
        ma_columns_list=params.ma_columns_list,
        columns_base=params.columns_base,
        lookahead=params.lookahead,
        n_back=params.n_back,
        training_and_testing=params.training_and_testing,
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
def prepare_data_flow(instrument: str, granularity: str, params_path: str | None = None) -> str:
    """Does NOT stop the SparkSession it gets/creates — a JVM only ever has one active
    SparkContext, so stopping it here would kill it out from under any other flow
    (prepare_all_flow, serve.py's retrain loop) or test fixture sharing the same
    process. Session lifecycle is the caller/process's responsibility; on a one-shot
    CLI run the JVM tears down naturally when the process exits."""
    params = load_params(params_path) if params_path else load_params()
    # Spark's stock default (1g driver memory) OOMs on real full-history production
    # data (verified against synthetic ~300-row test data only, which never exercised
    # this) -- the windowing/moving-average feature engineering materializes array
    # columns per row across the whole pair's history in the driver JVM. The original
    # prepare-training-and-inference-data.ipynb (this flow's notebook predecessor) set
    # driver/executor memory and maxResultSize to 70G explicitly -- that config simply
    # got dropped during the port to a package, not something nobody had considered.
    spark = (
        SparkSession.builder.appName("forex-ml-prepare-data")
        .config("spark.driver.memory", "70g")
        .config("spark.executor.memory", "70g")
        .config("spark.driver.maxResultSize", "70g")
        .getOrCreate()
    )
    pdf = pull_candles_task(instrument, granularity, params.feature)
    return engineer_and_save_task(spark, pdf, instrument, granularity, params.feature)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Stage-1 features for one (instrument, granularity) pair.")
    parser.add_argument("--instrument", required=True, help="e.g. EUR/USD")
    parser.add_argument("--granularity", required=True, help="e.g. H1")
    parser.add_argument("--params", default=None, help="Path to params.yaml (default: repo root)")
    args = parser.parse_args()
    prepare_data_flow(args.instrument, args.granularity, args.params)


if __name__ == "__main__":
    main()
