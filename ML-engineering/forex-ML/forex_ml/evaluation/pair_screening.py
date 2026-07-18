"""Formalizes this project's pair-screening protocol -- the sequence of checks
worked out by hand, pair by pair, across gold and a dozen currency crosses: pool a
GBT classifier's results across 5 sliding-window folds (never trust a single
train/test split), test several confidence thresholds at once and correct for
testing more than one, and only call a result real once it clears that bar.

One hard lesson from applying this by hand motivates NOT gating on a single-window
read or a regime-shift check before running the full protocol: AUD/USD's baseline
looked like the more credible candidate (beat both baselines, no train/test
distribution shift) and came back a clean, strong NEGATIVE under the full 5-fold
check, while USD/JPY looked weaker going in (lost to the persistence baseline, a
real moderate distribution shift) and came back with every one of 5 confidence
thresholds surviving correction. Single-window diagnostics narrow down false
positives; they do not reliably predict the multi-window verdict either way. So
this module's verdict is decided ONLY by the pooled multi-window result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest, ttest_1samp, wilcoxon


def calibrate_symmetric_barrier_pct(
    price: np.ndarray,
    holding_bars: int,
    round_to: float = 0.05,
) -> float:
    """Median absolute percent move over `holding_bars` bars, rounded to the
    nearest `round_to` -- this project's convention for a first-pass
    profit_take_pct/stop_loss_pct (used symmetrically) for a pair that hasn't been
    calibrated by hand yet, rather than blindly reusing another pair's threshold.
    `price` should be one pair's `mid_close` series, sorted chronologically."""
    if len(price) <= holding_bars:
        raise ValueError(f"Need more than holding_bars={holding_bars} rows, got {len(price)}")
    pct_moves = 100.0 * np.abs(price[holding_bars:] - price[:-holding_bars]) / price[:-holding_bars]
    median_move = float(np.median(pct_moves))
    return round(median_move / round_to) * round_to


@dataclass
class PooledThresholdResult:
    """One confidence threshold's result, pooled across every fold's test rows."""

    min_confidence: float
    total_trades: int
    pooled_win_rate: float
    win_rate_p_value: float  # binomial test, H1: pooled_win_rate > 0.5
    total_net_pnl_pct: float
    per_trade_t_p_value: float  # H1: mean per-trade net P&L > 0
    per_trade_wilcoxon_p_value: float  # H1: per-trade net P&L distribution > 0
    per_fold_win_rates: list[float]


def pool_fold_results(min_confidence: float, fold_simulation_results: list) -> PooledThresholdResult:
    """`fold_simulation_results`: one `forex_strategy.backtest.SimulationResult` per
    fold, all at the same `min_confidence`. Pools trade counts/win rate/net P&L
    across folds and computes both the win-rate significance test (the "wins more
    often" story) and the per-trade significance tests (the "wins bigger, even if
    not more often" story -- the shape that mattered for USD/CHF's ratio variant)."""
    total_trades = sum(r.n_trades for r in fold_simulation_results)
    total_wins = sum(round(r.win_rate * r.n_trades) for r in fold_simulation_results if r.n_trades > 0)
    total_net_pnl = sum(r.net_pnl_pct for r in fold_simulation_results)
    pooled_win_rate = total_wins / total_trades if total_trades > 0 else float("nan")
    win_rate_p = (
        binomtest(total_wins, total_trades, 0.5, alternative="greater").pvalue if total_trades > 0 else float("nan")
    )
    per_fold_win_rates = [r.win_rate for r in fold_simulation_results]

    pooled_per_trade = np.concatenate([r.per_trade_net_pnl_pct for r in fold_simulation_results])
    if len(pooled_per_trade) > 1:
        t_p = ttest_1samp(pooled_per_trade, 0.0, alternative="greater").pvalue
        w_p = wilcoxon(pooled_per_trade, alternative="greater").pvalue
    else:
        t_p = w_p = float("nan")

    return PooledThresholdResult(
        min_confidence=min_confidence,
        total_trades=total_trades,
        pooled_win_rate=pooled_win_rate,
        win_rate_p_value=win_rate_p,
        total_net_pnl_pct=total_net_pnl,
        per_trade_t_p_value=t_p,
        per_trade_wilcoxon_p_value=w_p,
        per_fold_win_rates=per_fold_win_rates,
    )


def benjamini_hochberg_pass(p_values: dict[float, float], alpha: float = 0.05) -> dict[float, bool]:
    """Standard BH step-up procedure across the thresholds tested at once (rather
    than picking a favorite threshold after seeing the results and testing it
    alone, which would overstate significance). Returns, per threshold key, whether
    it survives correction."""
    m = len(p_values)
    ranked = sorted(p_values.items(), key=lambda kv: kv[1])
    survives = {}
    # BH: find the largest rank k such that p_(k) <= (k/m)*alpha; every threshold
    # at or below that rank survives (not just ones individually under their own
    # critical value) -- this loop applies that step-up rule directly.
    largest_surviving_rank = 0
    for rank, (_, p) in enumerate(ranked, start=1):
        if p <= (rank / m) * alpha:
            largest_surviving_rank = rank
    for rank, (threshold, _) in enumerate(ranked, start=1):
        survives[threshold] = rank <= largest_surviving_rank
    return survives


@dataclass
class ScreeningVerdict:
    passed: bool
    passing_thresholds: list[float]
    reason: str
    threshold_results: dict[float, PooledThresholdResult]


def evaluate_screening_verdict(threshold_results: dict[float, PooledThresholdResult], alpha: float = 0.05) -> ScreeningVerdict:
    """A threshold passes if EITHER its win-rate story (wins more often than 50%)
    OR its per-trade story (wins bigger on average, even at <=50% win rate -- the
    payoff-asymmetry shape) survives BH correction across all thresholds tested,
    AND the pooled net P&L at that threshold is actually positive (a
    BH-significant-but-net-negative threshold, e.g. AUD/USD-like instability
    dressed up by a good p-value in one sub-test, should not pass). The pair
    passes overall if at least one threshold passes."""
    win_rate_pvals = {t: r.win_rate_p_value for t, r in threshold_results.items()}
    per_trade_pvals = {t: r.per_trade_t_p_value for t, r in threshold_results.items()}
    win_rate_survives = benjamini_hochberg_pass(win_rate_pvals, alpha)
    per_trade_survives = benjamini_hochberg_pass(per_trade_pvals, alpha)

    passing = [
        t for t, r in threshold_results.items()
        if r.total_net_pnl_pct > 0 and (win_rate_survives[t] or per_trade_survives[t])
    ]
    if passing:
        return ScreeningVerdict(
            passed=True, passing_thresholds=sorted(passing),
            reason="At least one confidence threshold cleared BH-corrected significance "
                   "(win-rate or per-trade) with positive pooled net P&L.",
            threshold_results=threshold_results,
        )
    return ScreeningVerdict(
        passed=False, passing_thresholds=[],
        reason="No confidence threshold cleared BH-corrected significance with positive "
               "pooled net P&L.",
        threshold_results=threshold_results,
    )
