"""Start a scheduled retraining deployment for the LSTM pipeline.

Usage:
    python -m forex_ml.flows.serve

Runs prepare -> split -> train for every (instrument, granularity) pair in
params.yaml, on a weekly schedule. Optional: most day-to-day use is just calling the
individual flows or `dvc repro` directly; this exists for unattended periodic
retraining, mirroring forex/flows/serve.py on the ETL side.
"""

from __future__ import annotations

from prefect import flow, serve

from forex_ml.config import load_params
from forex_ml.flows.prepare_data_flow import prepare_data_flow
from forex_ml.flows.split_flow import split_flow
from forex_ml.flows.train_flow import train_flow


@flow(name="forex-ml-retrain-all", log_prints=True)
def retrain_all_flow(params_path: str | None = None) -> None:
    params = load_params(params_path) if params_path else load_params()
    for instrument in params.feature.instruments:
        for granularity in params.feature.granularities:
            prepare_data_flow(instrument, granularity, params_path)
            split_flow(instrument, granularity, params_path)
            train_flow(instrument, granularity, params_path)


weekly_retrain = retrain_all_flow.to_deployment(
    name="weekly-retrain",
    cron="0 6 * * 1",  # Monday 06:00 UTC
)

if __name__ == "__main__":
    # to_deployment()'s stub type is ambiguous between the sync/async overloads for
    # mypy even though this call is always sync; narrow it explicitly.
    serve(weekly_retrain)  # type: ignore[arg-type]
