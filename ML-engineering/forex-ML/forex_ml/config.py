"""Validated pipeline configuration, loaded from params.yaml.

Replaces the raw dict literals that used to live inline in the notebooks and
lstm.py. Anything the pipeline reads at runtime should come from here rather
than a hand-edited dict, so a bad edit fails fast at load time instead of
silently producing a shape mismatch three stages later.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator

DEFAULT_PARAMS_PATH = Path(__file__).resolve().parent.parent / "params.yaml"


class FeatureParams(BaseModel):
    instruments: list[str]
    granularities: list[str]
    n_back: int
    lookahead: int
    ma_lookback_list: list[int]
    columns_base: list[str]
    ma_columns_list: list[str]
    training_and_testing: bool
    min_training_timestamp: datetime
    output_dir: str

    @property
    def engineered_columns(self) -> set[str]:
        """Every feature column name that Stage 1 (data/features.py) will produce."""
        columns = {
            "day_sin", "day_cos", "week_sin", "week_cos",
            "is_tokyo_session", "is_london_session", "is_new_york_session", "is_london_new_york_overlap",
            "volatility", "return", "diff_spread_close", "diff_volume",
        }
        for lookback in self.ma_lookback_list:
            for column in self.ma_columns_list:
                columns.add(f"{column}_MA_{lookback}")
        return columns


class SplitParams(BaseModel):
    """The training target is triple-barrier labeling (see
    forex_ml/data/triple_barrier.py): label each row by whichever of a profit-take,
    stop-loss, or max-holding-period (vertical) barrier is hit first, net of a
    round-trip spread charge and swap/rollover for any 5pm-New-York boundary
    actually crossed. There's no column_y/class_cutoff_percentiles here anymore --
    the label is computed directly as {-1, 0, +1}, not a continuous value that
    needs train-quantile thresholds to discretize.
    """

    profit_take_pct: float
    stop_loss_pct: float
    max_holding_bars: int
    swap_cost_pct_per_night: float = 0.0
    columns_x: list[str]
    train_val_proportion: list[float]

    @model_validator(mode="after")
    def _check_proportions(self) -> "SplitParams":
        if len(self.train_val_proportion) != 2:
            raise ValueError("train_val_proportion must have exactly 2 entries (train, val) — test is the remainder")
        if sum(self.train_val_proportion) >= 1.0:
            raise ValueError("train_val_proportion entries must sum to < 1.0 so a non-empty test split remains")
        return self


class TrainParams(BaseModel):
    number_of_cells_per_rnn_layer: list[int]
    number_of_cells_per_dense_layer: list[int]
    lstm_activation_function: str
    dense_activation_function: str
    final_dense_activation_function: str
    epochs: int
    batch_size: int
    learning_rate: float
    loss_function: str
    metrics: list[str]
    l1_regularization_constant: float
    l2_regularization_constant: float
    batch_normalization_momentum: float
    dense_dropout_rate: float
    rnn_dropout_rate: float
    rnn_recurrent_dropout_rate: float
    reduce_lr_on_plateau_factor: float
    reduce_lr_on_plateau_patience: int
    early_stopping_patience: int
    gradient_clip_norm: float = 1.0
    input_clip_value: float = 10.0
    tensorflow_seed: int
    mlflow_experiment_name: str
    mlflow_tracking_uri: str


class PipelineParams(BaseModel):
    feature: FeatureParams
    split: SplitParams
    train: TrainParams

    @model_validator(mode="after")
    def _check_split_columns_exist(self) -> "PipelineParams":
        unknown = set(self.split.columns_x) - self.feature.engineered_columns
        if unknown:
            raise ValueError(
                f"split.columns_x references columns Stage 1 never produces: {sorted(unknown)}. "
                f"Available columns: {sorted(self.feature.engineered_columns)}"
            )
        return self


def load_params(path: str | Path = DEFAULT_PARAMS_PATH) -> PipelineParams:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return PipelineParams(**raw)
