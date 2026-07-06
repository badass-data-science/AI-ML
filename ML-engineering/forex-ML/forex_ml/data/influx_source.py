"""Pull forward-filled candlestick data from InfluxDB.

Thin wrapper around the existing forex-etl / python-tools-and-shortcuts packages
(installed as real dependencies via pyproject.toml — see [tool.uv.sources] — rather
than the sys.path.append('/home/emily/...') hack the original notebooks used).
Credentials are handled entirely by forex.etl.config.database_config, which already
lazy-loads them from AWS Secrets Manager; nothing here touches secrets directly.
"""

from __future__ import annotations

import datetime

import pandas as pd
from forex.etl.config.database_config import INFLUXDB_ORG, INFLUXDB_TOKEN, INFLUXDB_URL
from python_tools_and_shortcuts.databases.influxdb.InfluxDbTool import InfluxDbTool

MEASUREMENT_NAME = "forward-filled candlestick"


def _make_ifc() -> InfluxDbTool:
    return InfluxDbTool(INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG)


def build_flux_query(
    measurement_name: str,
    instrument: str,
    granularity: str,
    min_timestamp: int,
    max_timestamp: int,
) -> str:
    return f'''
        start_s = {min_timestamp}
        stop_s = {max_timestamp}

        from(bucket: "forex")
          |> range(
              start: time(v: int(v: start_s) * 1000000000),
              stop: time(v: int(v: stop_s) * 1000000000)
            )
          |> filter(fn: (r) => r._measurement == "{measurement_name}")
          |> filter(fn: (r) => r.granularity == "{granularity}")
          |> filter(fn: (r) => r.instrument == "{instrument}")
          |> pivot(
              rowKey: ["_time"],
              columnKey: ["_field"],
              valueColumn: "_value"
          )
          |> drop(columns: ["_start", "_stop", "_measurement"])
        '''


def pull_candles(
    instrument: str,
    granularity: str,
    min_timestamp: int,
    max_timestamp: int,
    columns_sort: list[str] | None = None,
) -> pd.DataFrame:
    """Pull forward-filled candles for one (instrument, granularity) pair as a pandas DataFrame."""
    columns_sort = columns_sort or ["instrument", "granularity", "unix_epoch_s"]
    query = build_flux_query(MEASUREMENT_NAME, instrument, granularity, min_timestamp, max_timestamp)

    ifc = _make_ifc()
    try:
        df = ifc.run_flux_query_on_forex_database_and_get_dataframe(query)
    finally:
        del ifc

    df.sort_values(by=columns_sort, inplace=True)
    return df


def training_timestamp_range(min_training_timestamp: datetime.datetime) -> tuple[int, int]:
    """(min, max) unix-epoch-second range for a training-and-testing pull: fixed start, now."""
    max_timestamp = int(datetime.datetime.now().timestamp())
    min_timestamp = int(min_training_timestamp.timestamp())
    return min_timestamp, max_timestamp


def inference_timestamp_range(seconds_in_period: int, n_back: int) -> tuple[int, int]:
    """(min, max) unix-epoch-second range for an inference pull: just enough trailing bars."""
    max_timestamp = int(datetime.datetime.now().timestamp())
    min_timestamp = max_timestamp - (seconds_in_period * 3 * n_back)
    return min_timestamp, max_timestamp
