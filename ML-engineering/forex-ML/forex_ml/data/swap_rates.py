"""Fetch real, live swap/rollover rates from InfluxDB, for use as
`triple_barrier.py`'s `swap_cost_pct_per_night` and `forex_strategy.backtest`'s
long/short swap-cost inputs -- replacing the `params.yaml` placeholder constant
that was used before this module existed.

Mirrors `forex_ml/data/influx_source.py`'s `_make_ifc()` / lazy-`database_config`
pattern exactly (see that module's docstring for why `database_config` is imported
as a module and resolved fresh inside each function, never at import time).

Real per-instrument long/short financing rates are ingested into InfluxDB's
`swap-rate` measurement by forex-etl's `SwapRateETL` (tag: `instrument`; fields:
`long_rate`, `short_rate` -- see `forex/etl/models.py::SwapRateRecord`), sourced
from OANDA's v20 `/instruments` endpoint. Two conversions are needed before that
raw value is usable as `swap_cost_pct_per_night`, both handled by
`_annual_rate_to_swap_cost_pct_per_night` below:

1. OANDA's `longRate`/`shortRate` are ANNUAL rates expressed as decimals (0.05 =
   5%/year) -- confirmed via OANDA's own v20 API docs -- not per-night. Dividing
   by 365 (a simple Actual/365 approximation; OANDA's exact internal day-count
   convention isn't published in enough detail to do better) converts to a daily
   percentage.
2. Sign flip: `triple_barrier.py`'s convention is "a positive `swap_cost_pct_per_night`
   is a real cost, subtracted from net return." A NEGATIVE OANDA rate means you're
   charged, so it becomes a POSITIVE cost; a POSITIVE OANDA rate means you're
   credited, so it becomes a NEGATIVE cost (a net gain). This formula applies
   independently to `long_rate` and `short_rate` -- they are independently-signed
   OANDA fields (e.g. USD/JPY has been observed with a positive `long_rate` and a
   negative `short_rate` at the same time), not one rate mirrored with a flip.

No special handling is needed for OANDA's real-world "triple-charge on Wednesday
to cover the weekend" convention: `count_rollovers_crossed`
(forex_ml/data/triple_barrier.py) already counts each individual calendar-day
5pm-New-York boundary crossed, including weekend days, which nets out to the same
total charge for a weekend hold as OANDA's own Wednesday-multiplier convention --
just attributed to different calendar days. A plain per-calendar-day average
composes correctly with it as-is.
"""

from __future__ import annotations

import logging

from forex.etl.config import database_config
from python_tools_and_shortcuts.databases.influxdb.InfluxDbTool import InfluxDbTool

logger = logging.getLogger(__name__)

_DAYS_PER_YEAR = 365.0


def _make_ifc() -> InfluxDbTool:
    return InfluxDbTool(database_config.INFLUXDB_URL, database_config.INFLUXDB_TOKEN, database_config.INFLUXDB_ORG)


def _annual_rate_to_swap_cost_pct_per_night(annual_rate_decimal: float) -> float:
    """See module docstring for the two conversions this applies."""
    return -1.0 * (annual_rate_decimal * 100.0) / _DAYS_PER_YEAR


def build_swap_rate_query(instrument: str, lookback: str = "-30d") -> str:
    """Most recent swap-rate snapshot for `instrument`, pivoted so `long_rate`/
    `short_rate` land as columns on one row rather than one row per field --
    matches influx_source.py's `build_flux_query` pivot shape, which is what keeps
    `run_flux_query_on_forex_database_and_get_dataframe`'s `unix_epoch_s`
    conversion correct (see InfluxDbTool.py). `lookback` is a relative Flux
    duration (not an absolute range like candle pulls use): this query only ever
    wants "whatever the latest snapshot is," and a generous 30-day window keeps it
    working even against sparse/occasional ingestion."""
    return f'''
        from(bucket: "forex")
          |> range(start: {lookback})
          |> filter(fn: (r) => r._measurement == "swap-rate")
          |> filter(fn: (r) => r.instrument == "{instrument}")
          |> pivot(
              rowKey: ["_time"],
              columnKey: ["_field"],
              valueColumn: "_value"
          )
          |> sort(columns: ["_time"], desc: true)
          |> limit(n: 1)
        '''


def fetch_current_swap_rates(instrument: str, lookback: str = "-30d") -> tuple[float, float] | None:
    """`(long_swap_cost_pct_per_night, short_swap_cost_pct_per_night)` for the most
    recent swap-rate snapshot of `instrument`, or `None` if no snapshot exists
    within `lookback` (e.g. ingestion hasn't run for this instrument yet). Raises
    on any real connectivity/query error rather than swallowing it -- callers that
    want a forgiving fallback should use `resolve_swap_cost_pct_per_night` below,
    not this function directly."""
    query = build_swap_rate_query(instrument, lookback)
    ifc = _make_ifc()
    try:
        df = ifc.run_flux_query_on_forex_database_and_get_dataframe(query)
    finally:
        del ifc

    if len(df) == 0 or "long_rate" not in df.columns or "short_rate" not in df.columns:
        return None

    long_rate_decimal = float(df["long_rate"].iloc[0])
    short_rate_decimal = float(df["short_rate"].iloc[0])
    return (
        _annual_rate_to_swap_cost_pct_per_night(long_rate_decimal),
        _annual_rate_to_swap_cost_pct_per_night(short_rate_decimal),
    )


def resolve_swap_cost_pct_per_night(
    instrument: str, fallback_long: float, fallback_short: float | None = None,
) -> tuple[float, float]:
    """Try a live swap-rate fetch for `instrument`; fall back to the given
    constant(s) if no snapshot exists yet or the fetch fails for any reason (e.g.
    InfluxDB unreachable) -- callers should never hard-fail just because live
    swap-rate data isn't available. `fallback_short` defaults to `fallback_long`
    if not given, for callers (like forex-ML's `SplitParams`) that only ever
    configured one constant under the old, pre-live-rate scheme."""
    if fallback_short is None:
        fallback_short = fallback_long

    try:
        fetched = fetch_current_swap_rates(instrument)
    except Exception:
        logger.warning(
            "Live swap-rate fetch failed for %s; falling back to configured constants "
            "(long=%s, short=%s)", instrument, fallback_long, fallback_short, exc_info=True,
        )
        return fallback_long, fallback_short

    if fetched is None:
        logger.warning(
            "No live swap-rate snapshot found for %s; falling back to configured constants "
            "(long=%s, short=%s)", instrument, fallback_long, fallback_short,
        )
        return fallback_long, fallback_short

    logger.info(
        "Using live swap rates for %s: long=%.6f%%/night, short=%.6f%%/night", instrument, fetched[0], fetched[1],
    )
    return fetched
