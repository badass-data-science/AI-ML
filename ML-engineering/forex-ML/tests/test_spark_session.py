from __future__ import annotations

from unittest.mock import patch

from forex_ml.spark_session import DEFAULT_SPARK_MEMORY, build_spark_session


def test_build_spark_session_applies_memory_to_all_three_configs():
    """Checked by intercepting the builder's accumulated options right before
    getOrCreate() would consume them, rather than actually creating a session --
    a JVM only ever has one active SparkContext, so calling getOrCreate() in a test
    process that already has one running (the `spark` fixture, shared across the
    whole test suite) would silently return the EXISTING session and ignore these
    config values entirely, making a real-session-based assertion unreliable."""
    captured = {}

    def fake_get_or_create(self):
        captured.update(self._options)
        return "sentinel-session"

    with patch("pyspark.sql.SparkSession.Builder.getOrCreate", fake_get_or_create):
        result = build_spark_session("forex-ml-test-spark-session", memory="3g")

    assert result == "sentinel-session"
    assert captured["spark.app.name"] == "forex-ml-test-spark-session"
    assert captured["spark.driver.memory"] == "3g"
    assert captured["spark.executor.memory"] == "3g"
    assert captured["spark.driver.maxResultSize"] == "3g"


def test_build_spark_session_defaults_to_70g():
    captured = {}

    def fake_get_or_create(self):
        captured.update(self._options)
        return "sentinel-session"

    with patch("pyspark.sql.SparkSession.Builder.getOrCreate", fake_get_or_create):
        build_spark_session("forex-ml-test-spark-session")

    assert captured["spark.driver.memory"] == DEFAULT_SPARK_MEMORY == "70g"


def test_default_spark_memory_is_70g():
    assert DEFAULT_SPARK_MEMORY == "70g"
