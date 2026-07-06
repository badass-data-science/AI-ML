"""Real (non-mocked) integration tests for the InfluxDB pull boundary.

Spins up a real InfluxDB 2.x container via Docker, seeds it with synthetic
'forward-filled candlestick' rows using the exact schema forex_ml.data.influx_source
queries, and pulls it back through the real Flux query + InfluxDbTool + pandas
conversion path — nothing on the DB boundary itself is mocked. A second test runs the
full Stage-1 Prefect flow (pull -> engineer features -> Parquet) against the same
container.

Marked `integration` and excluded from the default `pytest` run (see pyproject.toml's
addopts) since it needs Docker and is slower than the unit suite. Run explicitly with:

    uv run pytest -v -m integration
"""

from __future__ import annotations

import datetime
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import numpy as np
import pytest
import yaml

from forex_ml.config import DEFAULT_PARAMS_PATH

pytestmark = pytest.mark.integration

INFLUXDB_IMAGE = "influxdb:2.7"
TEST_ORG = "forex-ml-test"
TEST_BUCKET = "forex"
TEST_TOKEN = "forex-ml-test-token-0123456789"  # local test container only, never a real secret

CANDLE_FIELDS = {
    "mid_open": float, "mid_high": float, "mid_low": float,
    "mid_close": float, "spread_close": float, "volume": float,
}
CANDLE_TAGS = frozenset({"instrument", "granularity"})


def _docker_available() -> bool:
    return shutil.which("docker") is not None


@pytest.fixture(scope="module")
def influxdb_container():
    if not _docker_available():
        pytest.skip("Docker is not available in this environment")

    name = f"forex-ml-test-influxdb-{uuid.uuid4().hex[:8]}"
    subprocess.run(
        [
            "docker", "run", "-d", "--rm",
            "--name", name,
            "-p", "8086",
            "-e", "DOCKER_INFLUXDB_INIT_MODE=setup",
            "-e", "DOCKER_INFLUXDB_INIT_USERNAME=test-admin",
            "-e", "DOCKER_INFLUXDB_INIT_PASSWORD=test-password-123",
            "-e", f"DOCKER_INFLUXDB_INIT_ORG={TEST_ORG}",
            "-e", f"DOCKER_INFLUXDB_INIT_BUCKET={TEST_BUCKET}",
            "-e", f"DOCKER_INFLUXDB_INIT_ADMIN_TOKEN={TEST_TOKEN}",
            INFLUXDB_IMAGE,
        ],
        check=True, capture_output=True, text=True,
    )
    try:
        port = subprocess.run(
            ["docker", "port", name, "8086/tcp"],
            check=True, capture_output=True, text=True,
        ).stdout.strip().split(":")[-1]
        url = f"http://localhost:{port}"

        for _ in range(60):
            try:
                if urllib.request.urlopen(f"{url}/health", timeout=1).status == 200:
                    break
            except (urllib.error.URLError, ConnectionError):
                pass
            time.sleep(1)
        else:
            raise RuntimeError("InfluxDB test container did not become healthy in time")

        yield {"url": url, "org": TEST_ORG, "bucket": TEST_BUCKET, "token": TEST_TOKEN}
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@pytest.fixture
def seeded_candles(influxdb_container):
    """Seed the test InfluxDB with synthetic 'forward-filled candlestick' rows for one
    (instrument, granularity) pair, using the real InfluxDbTool write path."""
    from python_tools_and_shortcuts.databases.influxdb.InfluxDbTool import InfluxDbTool

    ifc = InfluxDbTool(influxdb_container["url"], influxdb_container["token"], influxdb_container["org"])

    n = 50
    rng = np.random.default_rng(7)
    start = int(datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc).timestamp())
    timestamps = start + np.arange(n) * 3600
    base_price = 1.10 + np.cumsum(rng.normal(0, 0.0005, size=n))

    records = []
    for i in range(n):
        mid_open = float(base_price[i])
        mid_close = float(base_price[i] + rng.normal(0, 0.0002))
        mid_high = max(mid_open, mid_close) + abs(float(rng.normal(0, 0.0002)))
        mid_low = min(mid_open, mid_close) - abs(float(rng.normal(0, 0.0002)))
        records.append({
            "measurement": "forward-filled candlestick",
            "tags": {"instrument": "EUR/USD", "granularity": "H1"},
            "fields": {
                "mid_open": mid_open,
                "mid_high": mid_high,
                "mid_low": mid_low,
                "mid_close": mid_close,
                "spread_close": float(abs(rng.normal(0.0001, 0.00002))),
                "volume": float(rng.integers(100, 1000)),
            },
            "time": int(timestamps[i]),
        })

    ifc.insert_dictionary_list(records, CANDLE_TAGS, CANDLE_FIELDS, influxdb_container["bucket"])

    # The write path is asynchronous relative to query availability under load; give it
    # a moment before the tests below query it back.
    time.sleep(2)
    return {"n": n, "timestamps": timestamps, "instrument": "EUR/USD", "granularity": "H1"}


def _patch_database_config(monkeypatch, influxdb_container) -> None:
    """database_config lazy-loads credentials via a module-level `__getattr__`
    triggered on attribute access. forex_ml.data.influx_source now references it as
    `database_config.INFLUXDB_URL` (module access, resolved fresh each call) rather
    than `from database_config import INFLUXDB_URL` (which would freeze the resolved
    value into influx_source's own namespace at IMPORT time — see
    forex_ml/data/influx_source.py's module docstring and
    tests/test_secrets_isolation.py for the real bug this used to cause: merely
    collecting a sibling test file that imports forex_ml.flows.prepare_data_flow was
    enough to eagerly resolve real AWS-backed credentials, and no later monkeypatch
    here could undo that already-executed import).

    This patch (and the deferred `import forex_ml...` that follows it in each test)
    still happens inside the test function rather than at this file's top level, as
    defensive practice — but the fix above means it's no longer load-bearing for
    correctness the way it used to be.

    Patching must target `database_config.get_secret` specifically, not
    `python_tools_and_shortcuts.aws.secrets_manager.get_secret` — database_config did
    `from ... import get_secret`, binding its own independent reference, so patching
    the original module's attribute after that import has no effect on it.
    """
    import forex.etl.config.database_config as database_config

    database_config._load_secret.cache_clear()
    monkeypatch.setattr(
        database_config,
        "get_secret",
        lambda secret_name, region_name="us-west-2": json.dumps({
            "INFLUXDB_URL": influxdb_container["url"],
            "INFLUXDB_TOKEN": influxdb_container["token"],
            "INFLUXDB_ORG": influxdb_container["org"],
            "INFLUXDB_BUCKET": influxdb_container["bucket"],
        }),
    )


def test_pull_candles_reads_back_real_seeded_data(monkeypatch, influxdb_container, seeded_candles):
    _patch_database_config(monkeypatch, influxdb_container)

    from forex_ml.data import influx_source

    min_ts = int(seeded_candles["timestamps"][0]) - 3600
    max_ts = int(seeded_candles["timestamps"][-1]) + 3600

    df = influx_source.pull_candles(seeded_candles["instrument"], seeded_candles["granularity"], min_ts, max_ts)

    assert len(df) == seeded_candles["n"]
    for col in ["mid_open", "mid_high", "mid_low", "mid_close", "spread_close", "volume", "unix_epoch_s"]:
        assert col in df.columns
    assert df["unix_epoch_s"].is_monotonic_increasing


def _write_small_params(tmp_path: Path, output_dir: Path) -> Path:
    raw = yaml.safe_load(DEFAULT_PARAMS_PATH.read_text(encoding="utf-8"))
    raw["feature"].update({
        "instruments": ["EUR/USD"],
        "granularities": ["H1"],
        "n_back": 10,
        "lookahead": 2,
        "ma_lookback_list": [3, 5],
        "min_training_timestamp": "2023-01-01T00:00:00",
        "output_dir": str(output_dir),
    })
    path = tmp_path / "small_params.yaml"
    path.write_text(yaml.dump(raw))
    return path


def test_prepare_data_flow_end_to_end_against_real_influxdb(
    monkeypatch, influxdb_container, seeded_candles, tmp_path, spark,
):
    _patch_database_config(monkeypatch, influxdb_container)

    from forex_ml.flows.prepare_data_flow import prepare_data_flow
    from forex_ml.paths import non_time_series_parquet_path, pair_key, time_series_parquet_path

    output_dir = tmp_path / "output"
    params_path = _write_small_params(tmp_path, output_dir)

    key = prepare_data_flow(seeded_candles["instrument"], seeded_candles["granularity"], str(params_path))
    assert key == pair_key(seeded_candles["instrument"], seeded_candles["granularity"], 10, 2)

    ts_path = time_series_parquet_path(output_dir, key)
    non_ts_path = non_time_series_parquet_path(output_dir, key)
    assert ts_path.exists()
    assert non_ts_path.exists()

    pdf_time_series = spark.read.parquet(str(ts_path)).toPandas()
    assert len(pdf_time_series) > 0
    assert "pd_lead" in pdf_time_series.columns
