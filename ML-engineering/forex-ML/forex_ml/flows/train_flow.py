"""Prefect flow: thin wrapper around forex_ml.training.train.run so training is
orchestrated the same way as the other stages.

Run ad-hoc:
    python -m forex_ml.flows.train_flow --instrument EUR/USD --granularity H1
"""

from __future__ import annotations

import argparse
import json

from prefect import flow, get_run_logger

from forex_ml.training.train import run as run_training


@flow(name="forex-ml-train", log_prints=True)
def train_flow(instrument: str, granularity: str, params_path: str | None = None) -> dict:
    logger = get_run_logger()
    test_results = run_training(instrument, granularity, params_path)
    logger.info("Test metrics for %s %s: %s", instrument, granularity, test_results)
    return test_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the LSTM regressor for one (instrument, granularity) pair.")
    parser.add_argument("--instrument", required=True, help="e.g. EUR/USD")
    parser.add_argument("--granularity", required=True, help="e.g. H1")
    parser.add_argument("--params", default=None, help="Path to params.yaml (default: repo root)")
    args = parser.parse_args()
    test_results = train_flow(args.instrument, args.granularity, args.params)
    print(json.dumps(test_results, indent=2))


if __name__ == "__main__":
    main()
