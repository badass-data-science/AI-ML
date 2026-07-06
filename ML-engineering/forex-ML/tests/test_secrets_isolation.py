"""Regression test for a real bug: importing forex_ml's InfluxDB-touching modules
must never trigger AWS Secrets Manager resolution on its own -- only actually
calling a function that needs a live connection should.

database_config lazy-loads credentials via a module-level __getattr__ triggered on
attribute access. `from database_config import INFLUXDB_URL` at another module's top
level fires that trigger immediately, at IMPORT time, binding the resolved secret
into that module's namespace once, permanently, for the life of the process. Since
pytest collection imports every test file before running anything (even ones a
marker will later deselect), and forex_ml/flows/prepare_data_flow.py imports
forex_ml/data/influx_source.py at its own top level, merely collecting
test_prepare_data_flow.py / test_autocorrelation.py / test_stationarity.py used to
be enough to eagerly resolve real AWS-backed credentials and freeze them into
influx_source's namespace -- and no later test monkeypatch of
database_config.get_secret could ever undo an already-executed import. This was
caught by discovering that a "flaky" integration test was actually silently
querying a real production InfluxDB instead of its intended local Docker container,
because a sibling test file's collection-time import had already frozen the real
credentials moments earlier in the same pytest session.

The fix: reference database_config as a module (`from forex.etl.config import
database_config`, then `database_config.INFLUXDB_URL`) and resolve attributes fresh
at the point of use, rather than importing the values themselves. This test encodes
that invariant directly rather than re-importing modules pytest has already cached
(which would be a no-op and prove nothing).
"""

from __future__ import annotations


def test_influx_source_does_not_freeze_secrets_at_import_time():
    from forex_ml.data import influx_source

    assert not hasattr(influx_source, "INFLUXDB_URL")
    assert not hasattr(influx_source, "INFLUXDB_TOKEN")
    assert not hasattr(influx_source, "INFLUXDB_ORG")
    assert hasattr(influx_source, "database_config")
