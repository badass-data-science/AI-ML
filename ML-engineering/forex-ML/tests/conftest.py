"""Shared pytest fixtures: a local SparkSession and synthetic candle data."""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("forex-ml-tests")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield spark
    spark.stop()


@pytest.fixture
def synthetic_candles() -> pd.DataFrame:
    """A small, deterministic hourly candle series for one (instrument, granularity)
    pair, long enough to exercise moving averages + lookahead targets."""
    n = 300
    rng = np.random.default_rng(42)
    start = int(datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc).timestamp())
    timestamps = start + np.arange(n) * 3600

    base_price = 1.10 + np.cumsum(rng.normal(0, 0.0005, size=n))
    mid_open = base_price
    mid_close = base_price + rng.normal(0, 0.0002, size=n)
    mid_high = np.maximum(mid_open, mid_close) + np.abs(rng.normal(0, 0.0002, size=n))
    mid_low = np.minimum(mid_open, mid_close) - np.abs(rng.normal(0, 0.0002, size=n))
    spread_close = np.abs(rng.normal(0.0001, 0.00002, size=n))
    volume = rng.integers(100, 1000, size=n).astype(float)

    return pd.DataFrame({
        "instrument": "EUR/USD",
        "granularity": "H1",
        "unix_epoch_s": timestamps,
        "mid_open": mid_open,
        "mid_high": mid_high,
        "mid_low": mid_low,
        "mid_close": mid_close,
        "spread_close": spread_close,
        "volume": volume,
    })
