"""Quick, linear-approximation screening for "which time series most impacts the
target" — a repeatable tool for evaluating a candidate column BEFORE spending a
~13-minutes-per-epoch LSTM run on it, and for reconsidering existing `columns_x`
entries if GPU memory becomes the binding constraint.

Four complementary techniques, cheapest and least rigorous first:

  1. **Cross-correlation function (CCF)** — for each candidate, correlation between
     the candidate `lag` bars in the past and the target now. Direct extension of
     `autocorrelation.py`'s ACF/PACF machinery to a candidate-vs-target pair instead
     of a series against itself. Cheapest, least assumptions, good first pass.
  2. **Pairwise Granger causality** — does a candidate's own history improve a linear
     forecast of the target beyond the target's own history? Answers "does this
     column's past carry predictive information," not "does this column cause the
     target" — the name is famously misleading. Needs stationary inputs (checked
     here via `stationarity.check_stationarity` and flagged, not silently trusted).
  3. **VAR + block-exogeneity Wald tests + forecast-error variance decomposition
     (FEVD)** — the properly multivariate version of #2. Pairwise Granger tests miss
     multicollinearity (e.g. the four session flags overlapping in what they
     explain); a VAR estimates every candidate's contribution jointly, controlling
     for the others. FEVD directly answers "what fraction of the target's
     forecast-error variance is attributable to each candidate" — the closest linear
     analogue of a feature-importance ranking for a time-series system.
  4. **Lasso-regularized lagged regression** — a single-equation distributed-lag
     model (target ~ candidate laggedcolumns) with L1 regularization, which induces
     sparsity directly: a candidate's coefficients getting shrunk to exactly zero
     across all its lags is a literal, automatic "drop this" signal. Uses
     TimeSeriesSplit for cross-validation, not the default k-fold, since ordinary
     k-fold would shuffle future folds into past ones when picking the regularization
     strength — the same chronological-ordering discipline as everywhere else in
     this pipeline.

All four are linear approximations, and say so explicitly in their docstrings: an
LSTM can exploit nonlinear and cross-feature structure none of these can see. This is
a floor on which candidates are worth a full training run, not a ceiling on what
could possibly matter — same caveat as the ACF/PACF module's relationship to `n_back`.

Multiple-comparisons discipline: #2 and #3 test every candidate column against the
target, which is exactly the "many hypotheses, one correction" situation
`forex_ml.evaluation.multiple_comparisons` already exists for — reused here
(`benjamini_hochberg_report`) rather than reporting raw per-candidate p-values, so
adding more candidates to check doesn't inflate the false-discovery rate silently.
"""

from __future__ import annotations

import argparse
import warnings
from typing import cast

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import InterpolationWarning, grangercausalitytests

from forex_ml.config import load_params
from forex_ml.diagnostics.stationarity import check_stationarity
from forex_ml.evaluation.multiple_comparisons import benjamini_hochberg_report
from forex_ml.paths import non_time_series_parquet_path, pair_key
from forex_ml.spark_session import DEFAULT_SPARK_MEMORY, build_spark_session


def load_target_and_candidates(
    spark: SparkSession,
    output_dir: str,
    instrument: str,
    granularity: str,
    n_back: int,
    lookahead: int,
    target_column: str,
    candidate_columns: list[str],
) -> pd.DataFrame:
    """Real Stage-1 output for a pair, target + candidates only, chronologically
    sorted. `candidate_columns` can be ANY column Stage 1 produces — not just the
    ones currently in `split.columns_x` — so a brand new candidate can be screened
    before it's ever added to params.yaml."""
    key = pair_key(instrument, granularity, n_back, lookahead)
    columns = [target_column, *candidate_columns]
    return cast(
        pd.DataFrame,
        spark.read.parquet(str(non_time_series_parquet_path(output_dir, key)))
        .orderBy("unix_epoch_s")
        .select("unix_epoch_s", *columns)
        .toPandas(),
    )


def load_cross_pair_target_and_candidates(
    spark: SparkSession,
    output_dir: str,
    target_instrument: str,
    granularity: str,
    n_back: int,
    lookahead: int,
    target_column: str,
    candidate_pairs: list[tuple[str, list[str]]],
) -> pd.DataFrame:
    """Like load_target_and_candidates, but candidates are drawn from OTHER
    instruments at the same granularity/n_back/lookahead -- e.g. does GBP/USD's
    return help predict EUR/USD's volatility_lead? No new ingestion needed: every
    major pair already flows through the same Stage 1 pipeline, each under its own
    (instrument, granularity, n_back, lookahead) key.

    `candidate_pairs` is a list of (instrument, columns). Candidate columns from a
    DIFFERENT instrument than the target are renamed
    "{instrument_with_underscores}__{column}" before joining, since every pair's
    Stage 1 output uses the same column names (e.g. "return" exists for every
    instrument) and would otherwise collide. Joined to the target on unix_epoch_s
    via an inner join -- different pairs can have slightly different available
    timestamps (e.g. differing forward-fill history), so this keeps only
    timestamps common to the target and every candidate pair, rather than assuming
    perfect alignment.
    """
    target_key = pair_key(target_instrument, granularity, n_back, lookahead)
    merged = cast(
        pd.DataFrame,
        spark.read.parquet(str(non_time_series_parquet_path(output_dir, target_key)))
        .orderBy("unix_epoch_s")
        .select("unix_epoch_s", target_column)
        .toPandas(),
    )

    for candidate_instrument, columns in candidate_pairs:
        candidate_key = pair_key(candidate_instrument, granularity, n_back, lookahead)
        candidate_df = cast(
            pd.DataFrame,
            spark.read.parquet(str(non_time_series_parquet_path(output_dir, candidate_key)))
            .orderBy("unix_epoch_s")
            .select("unix_epoch_s", *columns)
            .toPandas(),
        )
        prefix = candidate_instrument.replace("/", "_")
        candidate_df = candidate_df.rename(columns={c: f"{prefix}__{c}" for c in columns})
        merged = merged.merge(candidate_df, on="unix_epoch_s", how="inner")

    return merged.sort_values("unix_epoch_s").reset_index(drop=True)


def cross_correlation_report(
    df: pd.DataFrame,
    target_column: str,
    candidate_columns: list[str],
    max_lag: int,
    practical_threshold: float = 0.1,
) -> dict[str, dict]:
    """corr(candidate[t-lag], target[t]) for lag=0..max_lag, per candidate — the
    same direction the model actually uses: history of a feature predicting the
    (already forward-looking) target realized at t. Reports the best (max |corr|)
    lag/magnitude, not just lag 0, since a candidate's most useful lag is rarely
    obvious in advance."""
    target = df[target_column]
    n = len(df)
    # approx 95% significance band for a correlation under the null of no relationship
    sig_threshold = 1.96 / np.sqrt(n)

    report = {}
    for column in candidate_columns:
        candidate = df[column]
        corrs = np.array([candidate.shift(lag).corr(target) for lag in range(max_lag + 1)])
        abs_corrs = np.abs(corrs)
        best_lag = int(np.nanargmax(abs_corrs))
        report[column] = {
            "lag0_corr": float(corrs[0]),
            "best_lag": best_lag,
            "best_corr": float(corrs[best_lag]),
            "practically_significant": bool(abs_corrs[best_lag] >= practical_threshold),
            "statistically_significant": bool(abs_corrs[best_lag] >= sig_threshold),
        }
    return report


def granger_causality_report(
    df: pd.DataFrame,
    target_column: str,
    candidate_columns: list[str],
    lag: int,
    alpha: float = 0.05,
) -> dict[str, dict]:
    """Pairwise Granger causality (does candidate's history improve a linear
    forecast of target beyond target's own history) at a single, SHARED lag across
    every candidate — deliberately not the minimum p-value across a scanned range of
    lags per candidate, which would be its own uncorrected multiple-comparisons
    problem layered on top of the one already being corrected for across candidates.
    Pass a lag informed by the target's own ACF/PACF diagnostic (autocorrelation.py)
    if you have one; otherwise a modest default is a reasonable start.

    Stationarity is a real assumption here, not a formality: a candidate/target pair
    that's actually non-stationary (see `verdict` from check_stationarity) can
    produce a spuriously "significant" Granger result. Checked and flagged per
    column rather than silently trusted.
    """
    p_values = {}
    stationarity_flags = {}
    for column in candidate_columns:
        pair_df = df[[target_column, column]].dropna()
        data = pair_df.to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            warnings.simplefilter("ignore", FutureWarning)
            result = grangercausalitytests(data, maxlag=[lag], verbose=False)
        p_values[column] = float(result[lag][0]["ssr_ftest"][1])
        stationarity_flags[column] = check_stationarity(pair_df[column].to_numpy())["verdict"]

    report = benjamini_hochberg_report(p_values, alpha=alpha)
    for column, flags in stationarity_flags.items():
        report[column]["candidate_stationarity"] = flags
    return report


def _lagged_design_rank_warning(data: pd.DataFrame, lag_order: int, rank_fraction_threshold: float = 0.9) -> str | None:
    """Builds the same lagged design VAR fits internally and checks its rank against
    its column count. A column count well above the actual rank means real,
    structural collinearity across lags (see var_fevd_report's docstring for the
    exact-sin/cos-lag case this exists to catch) -- not sample noise, which would
    show up as a poorly-conditioned but still full-rank matrix, not a rank drop."""
    lagged = {}
    for column in data.columns:
        for lag in range(lag_order + 1):
            lagged[f"{column}_lag{lag}"] = data[column].shift(lag)
    design = pd.DataFrame(lagged).dropna().to_numpy()
    rank = np.linalg.matrix_rank(design)
    n_columns = design.shape[1]
    if rank < rank_fraction_threshold * n_columns:
        return (
            f"lagged design matrix rank ({rank}) is well below its column count ({n_columns}) -- "
            f"likely structural collinearity across lags (e.g. a fixed-period sin/cos pair), "
            f"not just sample noise. Causality p-values for affected candidates may be unreliable "
            f"even if their FEVD share looks large."
        )
    return None


def var_fevd_report(
    df: pd.DataFrame,
    target_column: str,
    candidate_columns: list[str],
    lag_order: int,
    horizon: int,
    alpha: float = 0.05,
) -> dict:
    """Multivariate version of Granger causality (block-exogeneity Wald tests,
    controlling for every other candidate jointly instead of testing pairs in
    isolation) plus forecast-error variance decomposition (FEVD) -- the fraction of
    the target's forecast-error variance attributable to each candidate at
    `horizon` steps ahead. The closest linear analogue of a feature-importance
    ranking for a time-series system, and the one that catches candidates whose
    apparent pairwise Granger significance was really just correlation with another,
    more informative candidate (the four session flags are the obvious risk here).

    A real, structural gotcha this function checks for explicitly rather than
    silently trusting: a FIXED-period sin/cos encoding (day_sin/day_cos, week_sin/
    week_cos) is exactly linearly dependent across lags -- sin(w(t-k)) is an EXACT
    linear combination of sin(wt) and cos(wt) for any fixed lag k (the angle-
    subtraction identity), so including many lags of such a pair in an OLS-based
    model like VAR adds zero information and can wreck numerical conditioning.
    Confirmed directly on real data: an 11-lag block of day_sin/day_cos alone has
    rank 2 of 22 columns, condition number ~1.6e12. When that happens, the
    block-exogeneity Wald test's p-value for that candidate (and possibly others
    sharing the same underlying cyclical subspace, like the session flags) is not
    reliable, even though FEVD can still show a large share for it -- `rank_warning`
    below flags this so it isn't silently misread as "this candidate doesn't matter."
    """
    data = df[[target_column, *candidate_columns]].dropna()
    model = VAR(data)
    results = model.fit(lag_order)

    p_values = {}
    for column in candidate_columns:
        test = results.test_causality(target_column, [column], kind="f")
        p_values[column] = float(test.pvalue)
    causality_report = benjamini_hochberg_report(p_values, alpha=alpha)

    fevd = results.fevd(horizon)
    target_idx = list(data.columns).index(target_column)
    # fevd.decomp shape: (n_vars, horizon, n_vars) -- [equation, step, contributor]
    contributions_at_horizon = fevd.decomp[target_idx, horizon - 1, :]
    fevd_report = {
        column: float(contributions_at_horizon[list(data.columns).index(column)])
        for column in candidate_columns
    }

    rank_warning = _lagged_design_rank_warning(data, lag_order)

    return {
        "causality": causality_report,
        "fevd_fraction_of_target_variance": fevd_report,
        "lag_order": lag_order,
        "horizon": horizon,
        "rank_warning": rank_warning,
    }


def lasso_importance_report(
    df: pd.DataFrame,
    target_column: str,
    candidate_columns: list[str],
    max_lag: int,
    n_splits: int = 5,
) -> dict[str, dict]:
    """Single-equation distributed-lag Lasso: target ~ every candidate at lags
    0..max_lag, L1-regularized so unhelpful lags get shrunk to exactly zero --
    a direct, automatic "drop this" signal rather than a p-value to interpret.

    Uses TimeSeriesSplit (not the sklearn default k-fold) to pick the
    regularization strength, for the same reason nothing else in this pipeline
    shuffles: ordinary k-fold would let later folds' information leak into
    picking the model that gets evaluated on earlier ones.

    Features are standardized before fitting -- Lasso penalizes raw coefficient
    magnitude, so an unscaled comparison between e.g. a 0/1 session flag and a
    small-decimal volatility feature would be comparing apples to oranges.
    """
    lagged = {}
    for column in candidate_columns:
        for lag in range(max_lag + 1):
            lagged[f"{column}_lag{lag}"] = df[column].shift(lag)
    design = pd.DataFrame(lagged)
    design[target_column] = df[target_column]
    design = design.dropna()

    y = design.pop(target_column).to_numpy()
    feature_names = list(design.columns)
    X = StandardScaler().fit_transform(design.to_numpy())

    model = LassoCV(cv=TimeSeriesSplit(n_splits=n_splits))
    model.fit(X, y)

    report = {}
    for column in candidate_columns:
        lag_coefs = [
            model.coef_[feature_names.index(f"{column}_lag{lag}")] for lag in range(max_lag + 1)
        ]
        best_lag = int(np.argmax(np.abs(lag_coefs)))
        report[column] = {
            "max_abs_coefficient": float(np.max(np.abs(lag_coefs))),
            "best_lag": best_lag,
            "coefficient_at_best_lag": float(lag_coefs[best_lag]),
            "all_lags_zeroed": bool(np.allclose(lag_coefs, 0.0)),
        }
    report["_alpha_selected"] = float(model.alpha_)
    return report


def analyze_feature_impact(
    spark: SparkSession,
    output_dir: str,
    instrument: str,
    granularity: str,
    n_back: int,
    lookahead: int,
    target_column: str,
    candidate_columns: list[str],
    ccf_max_lag: int = 50,
    granger_lag: int = 10,
    var_lag_order: int = 10,
    var_horizon: int = 20,
    lasso_max_lag: int = 10,
) -> dict:
    """Runs all four techniques against one pair's real Stage-1 output."""
    df = load_target_and_candidates(
        spark, output_dir, instrument, granularity, n_back, lookahead, target_column, candidate_columns,
    )
    return {
        "n_observations": len(df),
        "cross_correlation": cross_correlation_report(df, target_column, candidate_columns, ccf_max_lag),
        "granger_causality": granger_causality_report(df, target_column, candidate_columns, granger_lag),
        "var_fevd": var_fevd_report(df, target_column, candidate_columns, var_lag_order, var_horizon),
        "lasso": lasso_importance_report(df, target_column, candidate_columns, lasso_max_lag),
    }


def analyze_cross_pair_feature_impact(
    spark: SparkSession,
    output_dir: str,
    target_instrument: str,
    granularity: str,
    n_back: int,
    lookahead: int,
    target_column: str,
    candidate_pairs: list[tuple[str, list[str]]],
    ccf_max_lag: int = 50,
    granger_lag: int = 10,
    var_lag_order: int = 10,
    var_horizon: int = 20,
    lasso_max_lag: int = 10,
) -> dict:
    """Runs all four techniques (see module docstring) against a target
    instrument's real Stage-1 output, with candidates drawn from OTHER
    instruments' Stage-1 output -- otherwise identical to analyze_feature_impact,
    since once the cross-pair candidates are loaded and renamed (see
    load_cross_pair_target_and_candidates), every downstream report function is
    candidate-source-agnostic."""
    df = load_cross_pair_target_and_candidates(
        spark, output_dir, target_instrument, granularity, n_back, lookahead, target_column, candidate_pairs,
    )
    candidate_columns = [
        f"{instrument.replace('/', '_')}__{column}"
        for instrument, columns in candidate_pairs
        for column in columns
    ]
    return {
        "n_observations": len(df),
        "cross_correlation": cross_correlation_report(df, target_column, candidate_columns, ccf_max_lag),
        "granger_causality": granger_causality_report(df, target_column, candidate_columns, granger_lag),
        "var_fevd": var_fevd_report(df, target_column, candidate_columns, var_lag_order, var_horizon),
        "lasso": lasso_importance_report(df, target_column, candidate_columns, lasso_max_lag),
    }


def _print_report(instrument: str, granularity: str, target_column: str, result: dict) -> None:
    print(f"{instrument} {granularity} — feature impact on {target_column}")
    print(f"  observations: {result['n_observations']}\n")

    print("  --- Cross-correlation (candidate[t-lag] vs target[t]) ---")
    for column, r in result["cross_correlation"].items():
        flag = "practical" if r["practically_significant"] else ("statistical only" if r["statistically_significant"] else "not significant")
        print(f"    {column:28s} best_lag={r['best_lag']:3d}  corr={r['best_corr']:+.3f}  ({flag})")

    print("\n  --- Granger causality (pairwise, BH-FDR corrected) ---")
    for column, r in result["granger_causality"].items():
        flag = "SIGNIFICANT" if r["significant_after_correction"] else "not significant"
        print(
            f"    {column:28s} p={r['p_value']:.4f}  p_adj={r['p_adjusted']:.4f}  {flag}  "
            f"(candidate stationarity: {r['candidate_stationarity']})"
        )

    var_result = result["var_fevd"]
    print(f"\n  --- VAR block-exogeneity + FEVD (lag_order={var_result['lag_order']}, horizon={var_result['horizon']}) ---")
    if var_result["rank_warning"]:
        print(f"    WARNING: {var_result['rank_warning']}")
    for column, r in var_result["causality"].items():
        flag = "SIGNIFICANT" if r["significant_after_correction"] else "not significant"
        fevd_frac = var_result["fevd_fraction_of_target_variance"][column]
        print(f"    {column:28s} p_adj={r['p_adjusted']:.4f}  {flag}  FEVD share={fevd_frac:.3f}")

    print(f"\n  --- Lasso lagged regression (alpha={result['lasso']['_alpha_selected']:.5f}) ---")
    for column, r in result["lasso"].items():
        if column == "_alpha_selected":
            continue
        status = "ALL LAGS ZEROED — drop candidate" if r["all_lags_zeroed"] else "retained"
        print(f"    {column:28s} best_lag={r['best_lag']:3d}  coef={r['coefficient_at_best_lag']:+.4f}  {status}")


def _parse_cross_pair_candidates(spec: str) -> list[tuple[str, list[str]]]:
    """Parses "GBP/USD:return,diff_spread_close;USD/JPY:volatility" into
    [("GBP/USD", ["return", "diff_spread_close"]), ("USD/JPY", ["volatility"])] --
    one semicolon-separated group per candidate instrument, columns within a group
    comma-separated."""
    pairs = []
    for group in spec.split(";"):
        instrument, columns = group.split(":")
        pairs.append((instrument.strip(), [c.strip() for c in columns.split(",")]))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick linear screening for which columns most impact the prediction target -- "
                     "a repeatable tool for evaluating candidate features before a full training run."
    )
    parser.add_argument("--instrument", required=True, help="e.g. EUR/USD")
    parser.add_argument("--granularity", required=True, help="e.g. H1")
    parser.add_argument(
        "--target", default="pd_lead",
        help="Column to screen candidates against (default: pd_lead). The training target is now "
             "triple-barrier labeling, computed at Stage 2 rather than selected from a named Stage-1 "
             "column, so there's no longer a single 'the configured target' to default to here -- "
             "pd_lead/volatility_lead/spread_close_lead remain valid Stage-1 reference columns.",
    )
    parser.add_argument(
        "--candidates", default=None,
        help="Comma-separated column names to evaluate (default: params.yaml's split.columns_x). "
             "Can be ANY column Stage 1 produces, including ones not yet in columns_x. "
             "Ignored if --cross-pair-candidates is given.",
    )
    parser.add_argument(
        "--cross-pair-candidates", default=None,
        help="Evaluate candidates from OTHER instruments instead of --instrument's own columns, e.g. "
             "'GBP/USD:return,diff_spread_close;USD/JPY:volatility' -- one semicolon-separated group per "
             "candidate instrument, columns within a group comma-separated. No new ingestion needed: every "
             "major pair already flows through Stage 1 under its own (instrument, granularity) key.",
    )
    parser.add_argument("--ccf-max-lag", type=int, default=50)
    parser.add_argument("--granger-lag", type=int, default=10, help="Shared lag for all pairwise Granger tests")
    parser.add_argument("--var-lag-order", type=int, default=10)
    parser.add_argument("--var-horizon", type=int, default=20)
    parser.add_argument("--lasso-max-lag", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--params", default=None, help="Path to params.yaml (default: repo root)")
    parser.add_argument(
        "--spark-memory", default=DEFAULT_SPARK_MEMORY,
        help=f"spark.driver.memory / spark.executor.memory / spark.driver.maxResultSize (default: {DEFAULT_SPARK_MEMORY})",
    )
    args = parser.parse_args()

    params = load_params(args.params) if args.params else load_params()
    spark = build_spark_session("forex-ml-feature-impact-diagnostic", memory=args.spark_memory)

    if args.cross_pair_candidates:
        candidate_pairs = _parse_cross_pair_candidates(args.cross_pair_candidates)
        result = analyze_cross_pair_feature_impact(
            spark, params.feature.output_dir, args.instrument, args.granularity,
            params.feature.n_back, params.feature.lookahead, args.target, candidate_pairs,
            ccf_max_lag=args.ccf_max_lag, granger_lag=args.granger_lag,
            var_lag_order=args.var_lag_order, var_horizon=args.var_horizon, lasso_max_lag=args.lasso_max_lag,
        )
    else:
        candidate_columns = args.candidates.split(",") if args.candidates else list(params.split.columns_x)
        result = analyze_feature_impact(
            spark, params.feature.output_dir, args.instrument, args.granularity,
            params.feature.n_back, params.feature.lookahead, args.target, candidate_columns,
            ccf_max_lag=args.ccf_max_lag, granger_lag=args.granger_lag,
            var_lag_order=args.var_lag_order, var_horizon=args.var_horizon, lasso_max_lag=args.lasso_max_lag,
        )
    _print_report(args.instrument, args.granularity, args.target, result)


if __name__ == "__main__":
    main()
