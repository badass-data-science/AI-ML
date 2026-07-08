"""CLI: load a registered forex-ML model's test-set predictions from MLflow, and run
the cost-aware backtest against them -- optionally with swap/rollover cost, the
5pm-New-York flatten rule, and volatility-gated position sizing from a second
(volatility_lead-trained) model for the same pair.

Run ad-hoc:
    python -m forex_strategy.run_backtest --tracking-uri sqlite:///../forex-ML/mlflow.db \
        --instrument EUR/USD --granularity H1
"""

from __future__ import annotations

import argparse
import tempfile

import numpy as np
from forex.eda.eda_config.eda_config import granularity_to_seconds_map
from mlflow.tracking import MlflowClient

from forex_strategy.backtest import (
    BacktestResult,
    position_size_from_predicted_volatility_class,
    predicted_classes_to_positions,
    simulate_trades,
)
from forex_strategy.model_registry import find_model_version, load_test_predictions


def _require_pd_lead(client: MlflowClient, run_id: str) -> None:
    column_y = client.get_run(run_id).data.params.get("column_y")
    if column_y != "pd_lead":
        raise ValueError(
            f"Run {run_id} was trained on column_y={column_y!r}, not 'pd_lead' -- "
            "this backtest assumes a directional target (see forex_strategy.backtest docstring)."
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
    use_volatility_sizing: bool = False,
    volatility_config_signature: str | None = None,
    size_by_class: tuple[float, float, float] = (1.0, 0.6, 0.3),
) -> BacktestResult:
    client = MlflowClient(tracking_uri=tracking_uri)
    resolved = find_model_version(
        client, registered_model_name, instrument, granularity, config_signature, column_y="pd_lead",
    )
    _require_pd_lead(client, resolved.run_id)  # belt-and-suspenders: tag lookup already filtered on it

    with tempfile.TemporaryDirectory() as tmp_dir:
        predictions = load_test_predictions(client, resolved.run_id, tmp_dir)

    positions = predicted_classes_to_positions(predictions["lstm_pred_proba"], min_confidence=min_confidence)

    entry_timestamp = predictions["test_timestamp"]
    exit_timestamp = None
    if swap_cost_pct_per_night != 0.0 or flatten_before_rollover:
        lookahead = int(client.get_run(resolved.run_id).data.params["lookahead"])
        granularity_seconds = float(granularity_to_seconds_map[granularity])
        exit_timestamp = entry_timestamp + lookahead * granularity_seconds

    position_size = None
    if use_volatility_sizing:
        vol_resolved = find_model_version(
            client, registered_model_name, instrument, granularity,
            volatility_config_signature, column_y="volatility_lead",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            vol_predictions = load_test_predictions(client, vol_resolved.run_id, tmp_dir)
        if not np.array_equal(vol_predictions["test_timestamp"], entry_timestamp):
            raise ValueError(
                "The pd_lead and volatility_lead models' test sets are not row-aligned by timestamp -- "
                "they must share the same n_back/lookahead/split configuration to be combined."
            )
        predicted_volatility_class = np.argmax(vol_predictions["lstm_pred_proba"], axis=1)
        position_size = position_size_from_predicted_volatility_class(predicted_volatility_class, size_by_class)

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
    parser.add_argument("--use-volatility-sizing", action="store_true")
    parser.add_argument("--volatility-config-signature", default=None)
    args = parser.parse_args()

    result = backtest_from_mlflow(
        args.tracking_uri, args.instrument, args.granularity,
        args.registered_model_name, args.config_signature, args.min_confidence,
        args.swap_cost_pct_per_night, args.flatten_before_rollover,
        args.use_volatility_sizing, args.volatility_config_signature,
    )
    print(f"rows={result.n_rows} trades={result.n_trades} flattened_for_rollover={result.n_flattened_for_rollover}")
    print(f"win_rate={result.win_rate:.3f}")
    print(f"gross_pnl_pct={result.gross_pnl_pct:.4f} cost_pct={result.cost_pct:.4f} net_pnl_pct={result.net_pnl_pct:.4f}")


if __name__ == "__main__":
    main()
