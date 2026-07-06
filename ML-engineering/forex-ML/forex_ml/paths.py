"""Shared output-path naming for every stage.

The original notebooks wrote Stage 1's Parquet output to a name keyed on
(instrument, granularity, n_back, lookahead), but Stage 2 always wrote to the single
shared path `output/data.pickled` regardless of which pair produced it — so only one
pair's prepared data could exist at a time before training. Every path below is keyed
on the pair so that no stage can silently overwrite another pair's output.
"""

from __future__ import annotations

from pathlib import Path


def pair_key(instrument: str, granularity: str, n_back: int, lookahead: int) -> str:
    safe_instrument = instrument.replace("/", "_")
    return f"{safe_instrument}__{granularity}__{n_back}__{lookahead}"


def stage1_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / "interim"


def stage2_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / "processed"


def time_series_parquet_path(output_dir: str | Path, key: str) -> Path:
    return stage1_dir(output_dir) / f"df_time_series__{key}.parquet"


def non_time_series_parquet_path(output_dir: str | Path, key: str) -> Path:
    return stage1_dir(output_dir) / f"df_non_time_series__{key}.parquet"


def stage1_config_path(output_dir: str | Path, key: str) -> Path:
    return stage1_dir(output_dir) / f"config__{key}.json"


def splits_npz_path(output_dir: str | Path, key: str) -> Path:
    return stage2_dir(output_dir) / f"splits__{key}.npz"
