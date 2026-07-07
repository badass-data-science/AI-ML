"""Shared Spark session construction for every CLI entry point that touches Spark.

Spark's stock default (1g driver memory) OOMs on real full-history production data
(verified against only ~300-row synthetic test data never exercises this) -- the
windowing/moving-average feature engineering and per-row array stacking materialize
array columns across a whole pair's history in the driver JVM. The original notebooks
this pipeline replaced hardcoded 70G-100G executor+driver memory for exactly this
reason, sized for one specific workstation -- carried forward here as an overridable
CLI default (`--spark-memory`, "70g") rather than a silently hardcoded value, since
the right amount depends on whatever machine is actually running it.
"""

from __future__ import annotations

from pyspark.sql import SparkSession

DEFAULT_SPARK_MEMORY = "70g"


def build_spark_session(app_name: str, memory: str = DEFAULT_SPARK_MEMORY) -> SparkSession:
    """One `memory` value applied to driver.memory, executor.memory, and
    driver.maxResultSize alike -- these three have moved together every time Spark
    memory sizing has come up in this project, and splitting them into separate
    knobs would be false precision without a concrete reason to size them
    differently."""
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.driver.memory", memory)
        .config("spark.executor.memory", memory)
        .config("spark.driver.maxResultSize", memory)
        .getOrCreate()
    )
