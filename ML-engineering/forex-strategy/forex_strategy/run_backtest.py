"""CLI: load a registered forex-ML model's test-set predictions from MLflow, and run
the spread-only backtest against them.

Run ad-hoc:
    python -m forex_strategy.run_backtest --tracking-uri sqlite:///../forex-ML/mlflow.db \
        --instrument EUR/USD --granularity H1
"""

from __future__ import annotations

import argparse
import tempfile

from mlflow.tracking import MlflowClient

from forex_strategy.backtest import BacktestResult, predicted_classes_to_positions, simulate_trades
from forex_strategy.model_registry import find_model_version, load_test_predictions


def backtest_from_mlflow(
    tracking_uri: str,
    instrument: str,
    granularity: str,
    registered_model_name: str = "forex-lstm",
    config_signature: str | None = None,
    min_confidence: float = 0.0,
) -> BacktestResult:
    client = MlflowClient(tracking_uri=tracking_uri)
    resolved = find_model_version(client, registered_model_name, instrument, granularity, config_signature)

    run = client.get_run(resolved.run_id)
    column_y = run.data.params.get("column_y")
    if column_y != "pd_lead":
        raise ValueError(
            f"Run {resolved.run_id} was trained on column_y={column_y!r}, not 'pd_lead' -- "
            "this backtest assumes a directional target (see forex_strategy.backtest docstring)."
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        predictions = load_test_predictions(client, resolved.run_id, tmp_dir)

    positions = predicted_classes_to_positions(predictions["lstm_pred_proba"], min_confidence=min_confidence)
    return simulate_trades(positions, predictions["test_y_raw"], predictions["test_spread"], predictions["test_price"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the spread-only backtest against a registered forex-ML model.")
    parser.add_argument("--tracking-uri", required=True, help="e.g. sqlite:///../forex-ML/mlflow.db")
    parser.add_argument("--instrument", required=True, help="e.g. EUR/USD")
    parser.add_argument("--granularity", required=True, help="e.g. H1")
    parser.add_argument("--registered-model-name", default="forex-lstm")
    parser.add_argument("--config-signature", default=None)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    args = parser.parse_args()

    result = backtest_from_mlflow(
        args.tracking_uri, args.instrument, args.granularity,
        args.registered_model_name, args.config_signature, args.min_confidence,
    )
    print(f"rows={result.n_rows} trades={result.n_trades} win_rate={result.win_rate:.3f}")
    print(f"gross_pnl_pct={result.gross_pnl_pct:.4f} cost_pct={result.cost_pct:.4f} net_pnl_pct={result.net_pnl_pct:.4f}")


if __name__ == "__main__":
    main()
