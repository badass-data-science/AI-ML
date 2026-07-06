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

_stack_time_series = F.udf(
    lambda *series: [[float(x) for x in s] for s in series],
    ArrayType(ArrayType(FloatType())),
)


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

    def split_train_val_test_by_proportion(self, train_val_proportion: list[float]) -> Splits:
        min_ts = self.df[self.timestamp_column].min()
        max_ts = self.df[self.timestamp_column].max()
        span = max_ts - min_ts

        train_stop = min_ts + int(train_val_proportion[0] * span)
        val_stop = train_stop + int(train_val_proportion[1] * span)

        df_train = self._slice_by_timestamp(min_ts, train_stop)
        df_val = self._slice_by_timestamp(train_stop, val_stop)
        df_test = self._slice_by_timestamp(val_stop, max_ts, inclusive_hi=True)

        percentiles = np.percentile(df_train[self.column_y].to_numpy(), self.class_cutoff_percentiles)
        df_train["outcome"] = self._compute_outcome(df_train[self.column_y].to_numpy(), percentiles)
        df_val["outcome"] = self._compute_outcome(df_val[self.column_y].to_numpy(), percentiles)
        df_test["outcome"] = self._compute_outcome(df_test[self.column_y].to_numpy(), percentiles)

        X_train = np.array([np.array(row) for row in df_train["X"].to_numpy()])
        X_val = np.array([np.array(row) for row in df_val["X"].to_numpy()])
        X_test = np.array([np.array(row) for row in df_test["X"].to_numpy()])

        # Normalize using train-only statistics, broadcast identically to val/test —
        # never let val/test distributions leak into the normalization.
        train_non_ts = (
            self.df_non_time_series[self.df_non_time_series[self.timestamp_column] < train_stop]
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
