"""CLI: load a registered forex-ML model's test-set predictions from MLflow, and run
the cost-aware backtest against them -- optionally with swap/rollover cost, the
5pm-New-York flatten rule, and realized-volatility-gated position sizing sourced
directly from the same model's own predictions (no second model to train/register/
keep row-aligned).

Run ad-hoc:
    python -m forex_strategy.run_backtest --tracking-uri sqlite:///../forex-ML/mlflow.db \
        --instrument EUR/USD --granularity H1
"""

from __future__ import annotations

import argparse
import tempfile

from forex.eda.eda_config.eda_config import granularity_to_seconds_map
from mlflow.tracking import MlflowClient

from forex_strategy.backtest import (
    BacktestResult,
    position_size_from_realized_volatility,
    predicted_classes_to_positions,
    simulate_trades,
)
from forex_strategy.model_registry import find_model_version, load_test_predictions


def _require_triple_barrier(client: MlflowClient, run_id: str) -> None:
    column_y = client.get_run(run_id).data.params.get("column_y")
    if column_y != "triple_barrier":
        raise ValueError(
            f"Run {run_id} was trained on column_y={column_y!r}, not 'triple_barrier' -- "
            "this backtest assumes triple-barrier labeling (see forex_ml.data.triple_barrier)."
        )


def backtest_from_mlflow(
    tracking_uri: str,
    instrument: str,
    granularity: str,
    registered_model_name: str = "forex-lstm",
    config_signature: str | None = None,
    min_confidence: float = 0.0,
    swap_cost_pct_per_night: float = 0.0,
    flatten_before_rollover: bool = False,
    size_by_realized_volatility: bool = False,
    target_volatility: float = 0.0005,
    max_position_size: float = 1.0,
) -> BacktestResult:
    client = MlflowClient(tracking_uri=tracking_uri)
    resolved = find_model_version(
        client, registered_model_name, instrument, granularity, config_signature, column_y="triple_barrier",
    )
    _require_triple_barrier(client, resolved.run_id)  # belt-and-suspenders: tag lookup already filtered on it

    with tempfile.TemporaryDirectory() as tmp_dir:
        predictions = load_test_predictions(client, resolved.run_id, tmp_dir)

    positions = predicted_classes_to_positions(predictions["lstm_pred_proba"], min_confidence=min_confidence)

    entry_timestamp = predictions["test_timestamp"]
    exit_timestamp = None
    if swap_cost_pct_per_night != 0.0 or flatten_before_rollover:
        granularity_seconds = float(granularity_to_seconds_map[granularity])
        exit_timestamp = entry_timestamp + predictions["test_exit_bar_offset"] * granularity_seconds

    position_size = None
    if size_by_realized_volatility:
        position_size = position_size_from_realized_volatility(
            predictions["test_realized_volatility"], target_volatility, max_position_size,
        )

    return simulate_trades(
        positions, predictions["test_y_raw"], predictions["test_spread"], predictions["test_price"],
        position_size=position_size,
        entry_timestamp=entry_timestamp, exit_timestamp=exit_timestamp,
        swap_cost_pct_per_night=swap_cost_pct_per_night, flatten_before_rollover=flatten_before_rollover,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the cost-aware backtest against a registered forex-ML model.")
    parser.add_argument("--tracking-uri", required=True, help="e.g. sqlite:///../forex-ML/mlflow.db")
    parser.add_argument("--instrument", required=True, help="e.g. EUR/USD")
    parser.add_argument("--granularity", required=True, help="e.g. H1")
    parser.add_argument("--registered-model-name", default="forex-lstm")
    parser.add_argument("--config-signature", default=None)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--swap-cost-pct-per-night", type=float, default=0.0)
    parser.add_argument("--flatten-before-rollover", action="store_true")
    parser.add_argument("--size-by-realized-volatility", action="store_true")
    parser.add_argument("--target-volatility", type=float, default=0.0005)
    parser.add_argument("--max-position-size", type=float, default=1.0)
    args = parser.parse_args()

    result = backtest_from_mlflow(
        args.tracking_uri, args.instrument, args.granularity,
        args.registered_model_name, args.config_signature, args.min_confidence,
        args.swap_cost_pct_per_night, args.flatten_before_rollover,
        args.size_by_realized_volatility, args.target_volatility, args.max_position_size,
    )
    print(f"rows={result.n_rows} trades={result.n_trades} flattened_for_rollover={result.n_flattened_for_rollover}")
    print(f"win_rate={result.win_rate:.3f}")
    print(f"gross_pnl_pct={result.gross_pnl_pct:.4f} cost_pct={result.cost_pct:.4f} net_pnl_pct={result.net_pnl_pct:.4f}")


if __name__ == "__main__":
    main()
