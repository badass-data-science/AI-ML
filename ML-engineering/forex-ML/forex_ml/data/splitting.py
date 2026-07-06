"""Stage 2: stack Stage-1 Parquet output into model-ready train/val/test tensors.

Refactor of prepare-ml-ts-data.ipynb. Loading + stacking the per-feature arrays into
one 'X' matrix column still needs Spark (that's what produced the Parquet in the first
place); everything after `.toPandas()` — the actual split math — is pure pandas/numpy
and fully unit-testable without a SparkSession.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, FloatType

def _stack_time_series_fn(*series: list[float]) -> list[list[float]]:
    """Stack per-feature n_back-length arrays into one (n_back, num_features) matrix
    per row — timesteps first, then features, matching the (timesteps, features)
    shape Keras LSTM layers expect.

    The original notebook's stack_time_series UDF built this the other way around
    ([[float(x) for x in s] for s in series], i.e. one row per FEATURE containing
    that feature's n_back values), producing (num_features, n_back) instead. That
    silently fed the LSTM a transposed representation — time and feature axes
    swapped relative to what `input_shape=(n_back, num_features)` in
    forex_ml/training/model.py assumes — and was undetected because Keras doesn't
    validate the semantic meaning of a shape, only its arithmetic compatibility.
    Caught here by actually running Stage 2 against real Spark output instead of
    hand-constructed test arrays already in the (assumed) correct shape.
    """
    n_back = len(series[0])
    return [[float(series[f][t]) for f in range(len(series))] for t in range(n_back)]


_stack_time_series = F.udf(_stack_time_series_fn, ArrayType(ArrayType(FloatType())))


def load_and_stack(
    spark: SparkSession,
    time_series_parquet_path: str,
    non_time_series_parquet_path: str,
    columns_x: list[str],
    column_y: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Stage-1 Parquet outputs and stack the per-feature arrays in `columns_x`
    order into a single 'X' matrix column per row. Column order here MUST match the
    `columns_x` order passed to TimeSeriesSplitter later — the X array's feature axis
    has no column names, only positions.
    """
    columns_non_time_series_to_keep = ["instrument", "granularity", "unix_epoch_s", *columns_x]

    df_time_series = spark.read.parquet(time_series_parquet_path)
    df_non_time_series = (
        spark.read.parquet(non_time_series_parquet_path)
        .select(*columns_non_time_series_to_keep)
        .orderBy("instrument", "granularity", "unix_epoch_s")
    )

    column_objects = [F.col(c) for c in columns_x]
    df_time_series = (
        df_time_series
        .withColumn("X", _stack_time_series(*column_objects))
        .select("instrument", "granularity", "unix_epoch_s", column_y, "X")
        .orderBy("instrument", "granularity", "unix_epoch_s")
    )

    pdf = (
        cast(pd.DataFrame, df_time_series.toPandas())
        .sort_values(by=["instrument", "granularity", "unix_epoch_s"]).reset_index(drop=True)
    )
    pdf_non_time_series = (
        cast(pd.DataFrame, df_non_time_series.toPandas())
        .sort_values(by=["instrument", "granularity", "unix_epoch_s"]).reset_index(drop=True)
    )
    return pdf, pdf_non_time_series


@dataclass
class Splits:
    train: dict[str, np.ndarray]
    val: dict[str, np.ndarray]
    test: dict[str, np.ndarray]

    def save_npz(self, path: str | Path) -> None:
        """Replaces the original pickle.dump(...) — .npz is a zip of plain .npy arrays,
        so loading it can't execute arbitrary code, and per-pair files stay small
        instead of one shared, unversioned 684MB pickle."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            train_M=self.train["M"], train_y=self.train["y"],
            val_M=self.val["M"], val_y=self.val["y"],
            test_M=self.test["M"], test_y=self.test["y"],
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "Splits":
        data = np.load(path)
        return cls(
            train={"M": data["train_M"], "y": data["train_y"]},
            val={"M": data["val_M"], "y": data["val_y"]},
            test={"M": data["test_M"], "y": data["test_y"]},
        )


class TimeSeriesSplitter:
    """Time-ordered train/val/test split, class discretization, and train-only
    normalization for one (instrument, granularity) pair.

    Refactor of the notebook's `Base` class: same math, pandas-only, no Spark
    dependency, fully unit-testable.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        df_non_time_series: pd.DataFrame,
        instrument: str,
        granularity: str,
        columns_x_components: list[str],
        timestamp_column: str = "unix_epoch_s",
        class_cutoff_percentiles: list[float] | None = None,
        column_y: str = "pd_lead",
    ) -> None:
        """`columns_x_components` MUST be the same list, in the same order, as the
        `columns_x` passed to `load_and_stack()` — it indexes into the stacked X
        matrix's feature axis positionally, not by name."""
        self.instrument = instrument
        self.granularity = granularity
        self.timestamp_column = timestamp_column
        self.column_y = column_y
        self.columns_x_components = columns_x_components
        self.class_cutoff_percentiles = list(class_cutoff_percentiles or [100.0 / 3.0, 200.0 / 3.0])

        self.df = (
            df[(df["instrument"] == instrument) & (df["granularity"] == granularity)]
            .copy().sort_values(by=timestamp_column).reset_index(drop=True)
        )
        self.df_non_time_series = (
            df_non_time_series[
                (df_non_time_series["instrument"] == instrument)
                & (df_non_time_series["granularity"] == granularity)
            ]
            .copy().sort_values(by=timestamp_column).reset_index(drop=True)
        )

    def _slice_by_timestamp(self, lo: float, hi: float, *, inclusive_hi: bool = False) -> pd.DataFrame:
        mask = (self.df[self.timestamp_column] >= lo) & (
            self.df[self.timestamp_column] <= hi if inclusive_hi else self.df[self.timestamp_column] < hi
        )
        return self.df[mask].copy().sort_values(by=self.timestamp_column).reset_index(drop=True)

    def _median_bar_spacing(self) -> float:
        timestamps = np.sort(self.df[self.timestamp_column].to_numpy())
        if len(timestamps) < 2:
            return 0.0
        return float(np.median(np.diff(timestamps)))

    def _compute_boundaries(
        self, train_val_proportion: list[float], purge_bars: int = 0,
    ) -> tuple[float, float, float, float, float, float]:
        """Returns (train_lo, train_hi, val_lo, val_hi, test_lo, test_hi).

        `purge_bars` carves a symmetric gap of `purge_bars` bars on both sides of
        each split boundary (a "purged" split, per Lopez de Prado's *Advances in
        Financial Machine Learning*). Without it, a training row sitting right at a
        boundary has a label computed via a forward lookahead into what is nominally
        validation data, and the first validation row's backward window reaches into
        training data. Neither is leakage in the sense of the model seeing future
        inputs at inference time — but the two adjacent rows are highly
        autocorrelated, which can optimistically bias the validation/test metric
        right at the seam. Pass `max(n_back, lookahead)` as `purge_bars` to remove
        exactly the rows capable of that overlap.
        """
        min_ts = self.df[self.timestamp_column].min()
        max_ts = self.df[self.timestamp_column].max()
        span = max_ts - min_ts

        train_stop = min_ts + int(train_val_proportion[0] * span)
        val_stop = train_stop + int(train_val_proportion[1] * span)
        purge_seconds = purge_bars * self._median_bar_spacing()

        return (
            min_ts, train_stop - purge_seconds,
            train_stop + purge_seconds, val_stop - purge_seconds,
            val_stop + purge_seconds, max_ts,
        )

    @staticmethod
    def _compute_outcome(y_values: np.ndarray, percentiles: np.ndarray) -> list[list[int]]:
        outcome = []
        for y in y_values:
            if y <= percentiles[0]:
                outcome.append([1, 0, 0])
            elif y <= percentiles[1]:
                outcome.append([0, 1, 0])
            else:
                outcome.append([0, 0, 1])
        return outcome

    def split_train_val_test_by_proportion(self, train_val_proportion: list[float], purge_bars: int = 0) -> Splits:
        train_lo, train_hi, val_lo, val_hi, test_lo, test_hi = self._compute_boundaries(
            train_val_proportion, purge_bars,
        )

        df_train = self._slice_by_timestamp(train_lo, train_hi)
        df_val = self._slice_by_timestamp(val_lo, val_hi)
        df_test = self._slice_by_timestamp(test_lo, test_hi, inclusive_hi=True)

        percentiles = np.percentile(df_train[self.column_y].to_numpy(), self.class_cutoff_percentiles)
        df_train["outcome"] = self._compute_outcome(df_train[self.column_y].to_numpy(), percentiles)
        df_val["outcome"] = self._compute_outcome(df_val[self.column_y].to_numpy(), percentiles)
        df_test["outcome"] = self._compute_outcome(df_test[self.column_y].to_numpy(), percentiles)

        X_train = np.array([np.array(row) for row in df_train["X"].to_numpy()])
        X_val = np.array([np.array(row) for row in df_val["X"].to_numpy()])
        X_test = np.array([np.array(row) for row in df_test["X"].to_numpy()])

        # Normalize using train-only statistics — cut off at the same purged boundary
        # as df_train itself, broadcast identically to val/test. Never let val/test
        # (or the purged gap) leak into the normalization.
        train_non_ts = (
            self.df_non_time_series[self.df_non_time_series[self.timestamp_column] < train_hi]
            [self.columns_x_components].to_numpy()
        )
        mean = np.mean(train_non_ts, axis=0)
        std = np.std(train_non_ts, axis=0)
        n_back = X_train.shape[1]
        mean_grid = np.tile(mean, (n_back, 1))
        std_grid = np.tile(std, (n_back, 1))

        X_train_norm = (X_train - mean_grid) / std_grid
        X_val_norm = (X_val - mean_grid) / std_grid
        X_test_norm = (X_test - mean_grid) / std_grid

        y_train = np.array([np.array(y) for y in df_train["outcome"].to_numpy()])
        y_val = np.array([np.array(y) for y in df_val["outcome"].to_numpy()])
        y_test = np.array([np.array(y) for y in df_test["outcome"].to_numpy()])

        return Splits(
            train={"M": X_train_norm, "y": y_train},
            val={"M": X_val_norm, "y": y_val},
            test={"M": X_test_norm, "y": y_test},
        )
