"""Prefect flow: Stage-1 Parquet -> stacked/split/normalized train/val/test tensors,
saved as .npz. Replaces prepare-ml-ts-data.ipynb.

Run ad-hoc:
    python -m forex_ml.flows.split_flow --instrument EUR/USD --granularity H1
"""

from __future__ import annotations

import argparse

from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE
from pyspark.sql import SparkSession

from forex_ml.config import SplitParams, load_params
from forex_ml.data.splitting import TimeSeriesSplitter, load_and_stack
from forex_ml.paths import non_time_series_parquet_path, pair_key, splits_npz_path, time_series_parquet_path


@task(name="load-split-and-save", cache_policy=NO_CACHE)
def load_split_and_save_task(
    # NO_CACHE: args include a SparkSession, which Prefect's default cache-key hashing
    # can't serialize — see prepare_data_flow.engineer_and_save_task for the same note.
    spark: SparkSession,
    instrument: str,
    granularity: str,
    n_back: int,
    lookahead: int,
    split_params: SplitParams,
    output_dir: str,
) -> str:
    logger = get_run_logger()
    key = pair_key(instrument, granularity, n_back, lookahead)

    pdf, pdf_non_time_series = load_and_stack(
        spark,
        str(time_series_parquet_path(output_dir, key)),
        str(non_time_series_parquet_path(output_dir, key)),
        split_params.columns_x,
        split_params.column_y,
    )

    splitter = TimeSeriesSplitter(
        pdf, pdf_non_time_series, instrument, granularity,
        columns_x_components=split_params.columns_x,
        class_cutoff_percentiles=split_params.class_cutoff_percentiles,
        column_y=split_params.column_y,
    )
    # purge_bars: a window can reach n_back bars backward and a label can reach
    # lookahead bars forward, so either direction can cross a split boundary — purge
    # max(n_back, lookahead) bars on both sides of each boundary to remove it.
    splits = splitter.split_train_val_test_by_proportion(
        split_params.train_val_proportion, purge_bars=max(n_back, lookahead),
    )

    out_path = splits_npz_path(output_dir, key)
    splits.save_npz(out_path)
    logger.info("Wrote splits for %s %s -> %s", instrument, granularity, out_path)
    return str(out_path)


@flow(name="forex-ml-split-data", log_prints=True)
def split_flow(instrument: str, granularity: str, params_path: str | None = None) -> str:
    """Does NOT stop the SparkSession — see prepare_data_flow's docstring for why."""
    params = load_params(params_path) if params_path else load_params()
    spark = SparkSession.builder.appName("forex-ml-split-data").getOrCreate()
    return load_split_and_save_task(
        spark, instrument, granularity,
        params.feature.n_back, params.feature.lookahead,
        params.split, params.feature.output_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Split/normalize Stage-1 output for one (instrument, granularity) pair.")
    parser.add_argument("--instrument", required=True, help="e.g. EUR/USD")
    parser.add_argument("--granularity", required=True, help="e.g. H1")
    parser.add_argument("--params", default=None, help="Path to params.yaml (default: repo root)")
    args = parser.parse_args()
    split_flow(args.instrument, args.granularity, args.params)


if __name__ == "__main__":
    main()
