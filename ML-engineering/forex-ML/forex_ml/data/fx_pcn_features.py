"""Reads forex-partial-correlation-network's (fx-pcn) local output and reduces it
to one pair's own daily structural features -- a genuinely different data SOURCE
from everything else in this pipeline (a sibling project's local run output, not
InfluxDB), so kept isolated in its own module.

Produces, one row per fx-pcn fit date:
- `fxpcn_network_mean_abs_partial_corr` -- the WHOLE NETWORK's average |partial
  correlation| that date (from fx-pcn's `density---...parquet`; same value for
  every pair on a given date -- this is the one metric this project has actually
  validated as predicting forward realized volatility, see this session's memory
  on validate_density_risk_signal.py/backtest_density_risk_overlay.py). NOT
  `density` (raw edge count) -- that metric was rejected as unreliable.
- `fxpcn_degree` -- how many of the other 6 majors THIS pair has a real
  (lasso-surviving) edge with that date, regardless of direction. A real,
  meaningful 0 for a date this pair has no edges at all, not a missing value.
- `fxpcn_directed_out_count` / `fxpcn_directed_in_count` -- of that degree, how
  many are this pair LEADING / FOLLOWING (an `i<->j` bidirected edge counts as
  both). From fx-pcn's edge table (`parameters---...parquet`); direction strings
  embed the real pair names (e.g. "USD/CHF->AUD/USD"), not a literal "i->j"
  placeholder -- see forex-ML/scripts/screen_lead_lag_edges.py's `_leading_edges`
  for the same lesson learned the hard way earlier this session.

Caller is responsible for the as-of join onto an H1 (or other sub-daily) frame
with the correct non-lookahead shift -- see
forex_ml.data.features.add_fx_pcn_features (mirrors add_daily_timeframe_features
exactly).
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd

FX_PCN_COLUMNS = [
    "fxpcn_network_mean_abs_partial_corr", "fxpcn_degree",
    "fxpcn_directed_out_count", "fxpcn_directed_in_count",
]


def _pair_direction_counts(edges: pd.DataFrame, pair: str) -> pd.DataFrame:
    """One row per date this `pair` has at least one edge, with degree/
    directed_out/directed_in counts. Dates with zero edges for this pair are
    NOT included here -- the caller reindexes against the full fit-date
    universe and fills those with a real 0, not a missing value."""
    involving = edges[(edges["pair_i"] == pair) | (edges["pair_j"] == pair)].copy()
    if involving.empty:
        return pd.DataFrame(columns=["date", "fxpcn_degree", "fxpcn_directed_out_count", "fxpcn_directed_in_count"])

    fwd = involving["pair_i"] + "->" + involving["pair_j"]  # pair_i leads pair_j
    rev = involving["pair_j"] + "->" + involving["pair_i"]  # pair_j leads pair_i
    bidir = involving["pair_i"] + "<->" + involving["pair_j"]
    pair_is_i = (involving["pair_i"] == pair).to_numpy()

    is_leader = np.where(pair_is_i, involving["direction"] == fwd, involving["direction"] == rev)
    is_follower = np.where(pair_is_i, involving["direction"] == rev, involving["direction"] == fwd)
    is_bidir = (involving["direction"] == bidir).to_numpy()
    involving["is_leader"] = is_leader | is_bidir
    involving["is_follower"] = is_follower | is_bidir

    grouped = involving.groupby("date")
    return pd.DataFrame({
        "fxpcn_degree": grouped.size(),
        "fxpcn_directed_out_count": grouped["is_leader"].sum(),
        "fxpcn_directed_in_count": grouped["is_follower"].sum(),
    }).reset_index()


def load_pair_fx_pcn_features(edges_path: str, density_path: str, pair: str) -> pd.DataFrame:
    """Returns [unix_epoch_s, fxpcn_network_mean_abs_partial_corr, fxpcn_degree,
    fxpcn_directed_out_count, fxpcn_directed_in_count], one row per fx-pcn fit
    date -- `unix_epoch_s` is midnight of that date, matching how
    prepare_data_flow.py's other pull_..._task functions hand timestamps to the
    as-of join in forex_ml.data.features."""
    # Deliberately NOT filtering out "undirected" edges here -- degree should
    # count every real (lasso-surviving) edge regardless of direction; only the
    # leader/follower classification inside _pair_direction_counts needs
    # direction to be resolved, and an "undirected" row already fails every one
    # of its fwd/rev/bidir string comparisons, correctly contributing 0 to
    # out/in counts while still correctly counting toward degree.
    edges = pd.read_parquet(edges_path, columns=["date", "pair_i", "pair_j", "direction"])
    density = pd.read_parquet(density_path, columns=["date", "mean_abs_partial_corr"])

    counts = _pair_direction_counts(edges, pair)
    # Every fx-pcn fit date is the universe -- a date with zero edges for this
    # pair is a real 0, not a gap, so this reindex+fillna is filling in TRUE
    # values, not guessing at missing ones.
    result = density.merge(counts, on="date", how="left")
    result[["fxpcn_degree", "fxpcn_directed_out_count", "fxpcn_directed_in_count"]] = (
        result[["fxpcn_degree", "fxpcn_directed_out_count", "fxpcn_directed_in_count"]].fillna(0.0)
    )
    result = result.rename(columns={"mean_abs_partial_corr": "fxpcn_network_mean_abs_partial_corr"})
    result["unix_epoch_s"] = result["date"].apply(
        lambda d: int(datetime.datetime.combine(d, datetime.time.min).timestamp())
    )
    return result[["unix_epoch_s", *FX_PCN_COLUMNS]].sort_values("unix_epoch_s").reset_index(drop=True)
