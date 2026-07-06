"""Prefect flow: run prepare_data_flow for every (instrument, granularity) pair listed
in params.yaml. Replaces make-all-training-data.ipynb's papermill loop.

Run:
    python -m forex_ml.flows.prepare_all_flow
"""

from __future__ import annotations

from prefect import flow, get_run_logger

from forex_ml.config import load_params
from forex_ml.flows.prepare_data_flow import prepare_data_flow


@flow(name="forex-ml-prepare-all", log_prints=True)
def prepare_all_flow(params_path: str | None = None) -> list[str]:
    logger = get_run_logger()
    params = load_params(params_path) if params_path else load_params()

    keys = []
    for instrument in params.feature.instruments:
        for granularity in params.feature.granularities:
            logger.info("Preparing %s %s", instrument, granularity)
            keys.append(prepare_data_flow(instrument, granularity, params_path))
    return keys


if __name__ == "__main__":
    prepare_all_flow()
