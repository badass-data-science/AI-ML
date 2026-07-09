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

from forex_ml.data.features import COLUMNS_PASSTHROUGH
from forex_ml.data.triple_barrier import triple_barrier_labels_from_frame

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
    columns_passthrough: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Stage-1 Parquet outputs and stack the per-feature arrays in `columns_x`
    order into a single 'X' matrix column per row. Column order here MUST match the
    `columns_x` order passed to TimeSeriesSplitter later — the X array's feature axis
    has no column names, only positions.

    No target column is selected here — the training target is computed by
    TimeSeriesSplitter itself via triple-barrier labeling (see that class's
    docstring), not selected from a Stage-1 column by name.

    `columns_passthrough` (default: COLUMNS_PASSTHROUGH, i.e. mid_close/spread_close)
    are carried through unchanged alongside X -- never fed to the model, but needed
    by TimeSeriesSplitter both to compute triple-barrier labels (mid_close/
    spread_close/unix_epoch_s) and to attach the test split's raw price/spread for
    backtesting.
    """
    columns_passthrough = list(COLUMNS_PASSTHROUGH) if columns_passthrough is None else columns_passthrough
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
        .select("instrument", "granularity", "unix_epoch_s", "X", *columns_passthrough)
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
    # The long/short swap costs actually used to compute this Splits' triple-
    # barrier labels (see forex_ml.data.swap_rates.resolve_swap_cost_pct_per_night)
    # -- persisted here, not re-derived later, because split_flow.py (which
    # produces this) and train.py's run() (which loads it back) are separate
    # process invocations that can happen hours or days apart. If train.py
    # independently re-resolved a live rate instead of reading these back, its
    # MLflow-logged swap costs could silently mismatch the values that actually
    # produced y/y_raw -- logging numbers with zero influence on the labels
    # they're supposedly describing. Each defaults to 0.0 independently for
    # pre-migration .npz files that don't have it (see load_npz).
    long_swap_cost_pct_per_night: float = 0.0
    short_swap_cost_pct_per_night: float = 0.0

    def save_npz(self, path: str | Path) -> None:
        """Replaces the original pickle.dump(...) — .npz is a zip of plain .npy arrays,
        so loading it can't execute arbitrary code, and per-pair files stay small
        instead of one shared, unversioned 684MB pickle.

        `test` carries six extra keys (timestamp/price/spread/y_raw/exit_bar_offset/
        realized_volatility) that train/val don't -- a backtest only ever needs to
        reconstruct P&L on the held-out test set, so train/val stay exactly
        {"M", "y"} rather than carrying reference data nothing reads.
        `exit_bar_offset` (how many bars the triple-barrier label actually took to
        resolve) lets a backtest compute a real, variable holding period instead of
        assuming a fixed one. `realized_volatility` (a fixed-window backward-looking
        reference, see COLUMNS_PASSTHROUGH) lets a backtest scale position size down
        as recent realized volatility rises, without needing a second, forward-
        looking volatility model.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            train_M=self.train["M"], train_y=self.train["y"],
            val_M=self.val["M"], val_y=self.val["y"],
            test_M=self.test["M"], test_y=self.test["y"],
            long_swap_cost_pct_per_night=self.long_swap_cost_pct_per_night,
            short_swap_cost_pct_per_night=self.short_swap_cost_pct_per_night,
            test_timestamp=self.test["timestamp"], test_price=self.test["price"], test_spread=self.test["spread"],
            test_y_raw=self.test["y_raw"], test_exit_bar_offset=self.test["exit_bar_offset"],
            test_realized_volatility=self.test["realized_volatility"],
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "Splits":
        data = np.load(path)
        return cls(
            train={"M": data["train_M"], "y": data["train_y"]},
            val={"M": data["val_M"], "y": data["val_y"]},
            test={
                "M": data["test_M"], "y": data["test_y"],
                "timestamp": data["test_timestamp"], "price": data["test_price"], "spread": data["test_spread"],
                "y_raw": data["test_y_raw"], "exit_bar_offset": data["test_exit_bar_offset"],
                "realized_volatility": data["test_realized_volatility"],
            },
            # Pre-migration .npz files (written before swap rates, or before the
            # long/short split, were wired in) don't have these keys -- each
            # defaults to 0.0 independently, matching the placeholder constant
            # that was in effect when they were written.
            long_swap_cost_pct_per_night=(
                float(data["long_swap_cost_pct_per_night"]) if "long_swap_cost_pct_per_night" in data else 0.0
            ),
            short_swap_cost_pct_per_night=(
                float(data["short_swap_cost_pct_per_night"]) if "short_swap_cost_pct_per_night" in data else 0.0
            ),
        )


class TimeSeriesSplitter:
    """Time-ordered train/val/test split, triple-barrier labeling, and train-only
    normalization for one (instrument, granularity) pair.

    Refactor of the notebook's `Base` class: same split/normalization math,
    pandas-only, no Spark dependency, fully unit-testable. The training target
    itself is computed here via `triple_barrier_labels_from_frame` (see
    forex_ml/data/triple_barrier.py) rather than selected from a pre-computed
    Stage-1 column: label each row by whichever of a profit-take, stop-loss, or
    max-holding-period barrier is hit first, net of round-trip spread and any
    swap/rollover for a 5pm-New-York boundary actually crossed. The label is
    already discrete ({-1, 0, +1}), so there's no percentile-threshold fitting
    step anymore -- see `_label_to_one_hot`.
    """

    _LABEL_TO_ONE_HOT = {-1: [1, 0, 0], 0: [0, 1, 0], 1: [0, 0, 1]}

    def __init__(
        self,
        df: pd.DataFrame,
        df_non_time_series: pd.DataFrame,
        instrument: str,
        granularity: str,
        columns_x_components: list[str],
        profit_take_pct: float,
        stop_loss_pct: float,
        max_holding_bars: int,
        long_swap_cost_pct_per_night: float = 0.0,
        short_swap_cost_pct_per_night: float = 0.0,
        timestamp_column: str = "unix_epoch_s",
    ) -> None:
        """`columns_x_components` MUST be the same list, in the same order, as the
        `columns_x` passed to `load_and_stack()` — it indexes into the stacked X
        matrix's feature axis positionally, not by name."""
        self.instrument = instrument
        self.granularity = granularity
        self.timestamp_column = timestamp_column
        self.columns_x_components = columns_x_components
        # Stashed so _build_splits can attach them to the Splits it returns -- see
        # Splits' long/short swap-cost docstring for why these need to survive
        # the split/train process boundary rather than being re-derived later.
        self.long_swap_cost_pct_per_night = long_swap_cost_pct_per_night
        self.short_swap_cost_pct_per_night = short_swap_cost_pct_per_night

        df_pair = (
            df[(df["instrument"] == instrument) & (df["granularity"] == granularity)]
            .copy().sort_values(by=timestamp_column).reset_index(drop=True)
        )
        # Computed ONCE across the full per-pair series, before any splitting --
        # mirrors how Stage 1 currently pre-computes pd_lead/volatility_lead once,
        # rather than per-split (which would let each split's boundary rows use a
        # slightly different labeling window). Shortens the frame by
        # max_holding_bars, the same shape of trimming pd_lead's fixed lookahead
        # window used to require.
        self.df = triple_barrier_labels_from_frame(
            df_pair,
            profit_take_pct=profit_take_pct, stop_loss_pct=stop_loss_pct,
            max_holding_bars=max_holding_bars,
            long_swap_cost_pct_per_night=long_swap_cost_pct_per_night,
            short_swap_cost_pct_per_night=short_swap_cost_pct_per_night,
            price_column="mid_close", spread_column="spread_close", timestamp_column=timestamp_column,
        )
        self.df_non_time_series = (
            df_non_time_series[
                (df_non_time_series["instrument"] == instrument)
                & (df_non_time_series["granularity"] == granularity)
            ]
            .copy().sort_values(by=timestamp_column).reset_index(drop=True)
        )

    @classmethod
    def _label_to_one_hot(cls, label_values: np.ndarray) -> list[list[int]]:
        return [cls._LABEL_TO_ONE_HOT[int(label)] for label in label_values]

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
        right at the seam. Pass `max(n_back, max_holding_bars)` as `purge_bars` to
        remove exactly the rows capable of that overlap.
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

    def _rolling_boundaries(
        self,
        n_folds: int,
        min_train_bars: int,
        val_bars: int,
        test_bars: int,
        window: str,
        purge_bars: int = 0,
    ) -> list[tuple[float, float, float, float, float, float]]:
        """Boundary timestamps for `n_folds` walk-forward folds — same contiguous
        train/val/test-block-plus-purge-gap structure as `_compute_boundaries`, but
        positioned at `n_folds` successive points along the timeline (a rolling
        robustness diagnostic) instead of just once.

        Every fold's val/test blocks are fixed-length (`val_bars`/`test_bars`) and
        slide forward by `test_bars` each fold — one test block is the walk-forward
        "step". `window` controls the training block:
          - "sliding": fixed length (`min_train_bars`), sliding forward with the
            fold — every fold trains on a comparably-recent, comparably-sized
            history. More robust to regime change; discards older data.
          - "expanding": always starts at the very first bar and grows by
            `test_bars` each fold, so later folds accumulate strictly more history.
            Uses all available data; assumes older data is as relevant as recent
            data.
        """
        if window not in ("sliding", "expanding"):
            raise ValueError(f"window must be 'sliding' or 'expanding', got {window!r}")

        bar_spacing = self._median_bar_spacing()
        if bar_spacing <= 0:
            raise ValueError("Cannot compute rolling folds: fewer than 2 rows in this pair's data")

        min_ts = self.df[self.timestamp_column].min()
        max_ts = self.df[self.timestamp_column].max()
        total_bars = (max_ts - min_ts) / bar_spacing

        bars_needed = min_train_bars + val_bars + test_bars + (n_folds - 1) * test_bars
        if bars_needed > total_bars:
            max_folds = max(0, int((total_bars - min_train_bars - val_bars - test_bars) // test_bars) + 1)
            raise ValueError(
                f"Not enough data for {n_folds} rolling fold(s): need ~{bars_needed:.0f} bars, "
                f"have ~{total_bars:.0f}. At most {max_folds} fold(s) fit with these window sizes — "
                f"reduce n_folds/min_train_bars/val_bars/test_bars, or wait for more data to accumulate."
            )

        purge_seconds = purge_bars * bar_spacing
        boundaries = []
        for fold in range(n_folds):
            if window == "expanding":
                train_start_bars = 0.0
                train_bars = min_train_bars + fold * test_bars
            else:
                train_start_bars = float(fold * test_bars)
                train_bars = min_train_bars

            train_lo = min_ts + train_start_bars * bar_spacing
            train_hi_raw = train_lo + train_bars * bar_spacing
            val_lo_raw = train_hi_raw
            val_hi_raw = val_lo_raw + val_bars * bar_spacing
            test_lo_raw = val_hi_raw
            test_hi_raw = test_lo_raw + test_bars * bar_spacing

            boundaries.append((
                train_lo, train_hi_raw - purge_seconds,
                val_lo_raw + purge_seconds, val_hi_raw - purge_seconds,
                test_lo_raw + purge_seconds, test_hi_raw,
            ))
        return boundaries

    def _build_splits(
        self, train_lo: float, train_hi: float, val_lo: float, val_hi: float, test_lo: float, test_hi: float,
    ) -> Splits:
        df_train = self._slice_by_timestamp(train_lo, train_hi)
        df_val = self._slice_by_timestamp(val_lo, val_hi)
        df_test = self._slice_by_timestamp(test_lo, test_hi, inclusive_hi=True)

        df_train["outcome"] = self._label_to_one_hot(df_train["label"].to_numpy())
        df_val["outcome"] = self._label_to_one_hot(df_val["label"].to_numpy())
        df_test["outcome"] = self._label_to_one_hot(df_test["label"].to_numpy())

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
            test={
                "M": X_test_norm, "y": y_test,
                # Raw (unnormalized) reference data for the test split only -- a
                # backtest needs the actual timestamp/price/spread a row traded at,
                # not the train-normalized feature values used for prediction. See
                # COLUMNS_PASSTHROUGH in forex_ml.data.features for how these survive
                # Stage 1 without becoming model input features.
                "timestamp": df_test[self.timestamp_column].to_numpy(),
                "price": df_test["mid_close"].to_numpy(),
                "spread": df_test["spread_close"].to_numpy(),
                # The pre-cost realized return at the row's actual exit bar, not just
                # which barrier it hit -- "outcome"/"y" tells you the model's
                # classification target, a backtest computing actual $ P&L needs the
                # realized magnitude. raw_return_pct (not net_return_pct, which is
                # already net of spread/swap) so a backtest charging its own cost
                # doesn't double-count it -- see triple_barrier.py's docstring.
                "y_raw": df_test["raw_return_pct"].to_numpy(),
                # How many bars this row's label actually took to resolve -- lets a
                # backtest compute a real, variable holding period (and therefore
                # accurate swap-cost accounting) instead of assuming a fixed one.
                "exit_bar_offset": df_test["exit_bar_offset"].to_numpy(),
                # Fixed-window, backward-looking realized volatility (see
                # COLUMNS_PASSTHROUGH) -- lets a backtest scale position size down as
                # recent realized volatility rises, without needing a second,
                # forward-looking volatility model.
                "realized_volatility": df_test["realized_volatility"].to_numpy(),
            },
            long_swap_cost_pct_per_night=self.long_swap_cost_pct_per_night,
            short_swap_cost_pct_per_night=self.short_swap_cost_pct_per_night,
        )

    def split_train_val_test_by_proportion(self, train_val_proportion: list[float], purge_bars: int = 0) -> Splits:
        return self._build_splits(*self._compute_boundaries(train_val_proportion, purge_bars))

    def rolling_folds(
        self,
        n_folds: int,
        min_train_bars: int,
        val_bars: int,
        test_bars: int,
        window: str = "sliding",
        purge_bars: int = 0,
    ) -> list[Splits]:
        """`n_folds` walk-forward train/val/test splits for a robustness diagnostic
        across time periods — see `_rolling_boundaries` for the "sliding" vs
        "expanding" window semantics. Each fold gets its own train-only
        normalization statistics (via `_build_splits`), exactly like a single split —
        no leakage across folds, since a later fold's training data was never
        touched by an earlier fold's stats."""
        boundaries = self._rolling_boundaries(n_folds, min_train_bars, val_bars, test_bars, window, purge_bars)
        return [self._build_splits(*b) for b in boundaries]
