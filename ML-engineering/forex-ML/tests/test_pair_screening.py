from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forex_ml.evaluation.pair_screening import (
    PooledThresholdResult,
    benjamini_hochberg_pass,
    calibrate_symmetric_barrier_pct,
    evaluate_screening_verdict,
    pool_fold_results,
)


def test_calibrate_symmetric_barrier_pct_matches_known_median_move():
    # Price doubles every `holding_bars` step in a perfectly regular way so the
    # median 24-bar move is exactly computable by hand.
    holding_bars = 4
    price = np.array([100.0, 101, 102, 103, 105.0, 106, 107, 108, 110.0])
    # moves: (105-100)/100=5%, (106-101)/101~=4.95%, ... last (110-105)/105~=4.76%
    result = calibrate_symmetric_barrier_pct(price, holding_bars, round_to=0.05)
    assert 4.5 < result < 5.5


def test_calibrate_symmetric_barrier_pct_rejects_too_few_rows():
    price = np.array([100.0, 101.0, 102.0])
    try:
        calibrate_symmetric_barrier_pct(price, holding_bars=24)
        assert False, "expected ValueError"
    except ValueError:
        pass


@dataclass
class _FakeSimulationResult:
    n_trades: int
    win_rate: float
    net_pnl_pct: float
    per_trade_net_pnl_pct: np.ndarray


def test_pool_fold_results_combines_trade_counts_and_pnl_across_folds():
    fold_a = _FakeSimulationResult(n_trades=100, win_rate=0.6, net_pnl_pct=10.0,
                                    per_trade_net_pnl_pct=np.full(100, 0.1))
    fold_b = _FakeSimulationResult(n_trades=50, win_rate=0.5, net_pnl_pct=-2.0,
                                    per_trade_net_pnl_pct=np.full(50, -0.04))

    pooled = pool_fold_results(0.50, [fold_a, fold_b])

    assert pooled.total_trades == 150
    assert pooled.total_net_pnl_pct == 8.0
    assert pooled.pooled_win_rate == (60 + 25) / 150
    assert pooled.per_fold_win_rates == [0.6, 0.5]
    assert 0.0 <= pooled.win_rate_p_value <= 1.0


def test_pool_fold_results_handles_zero_trades():
    fold = _FakeSimulationResult(n_trades=0, win_rate=0.0, net_pnl_pct=0.0, per_trade_net_pnl_pct=np.array([]))
    pooled = pool_fold_results(0.55, [fold])
    assert pooled.total_trades == 0
    assert pooled.pooled_win_rate != pooled.pooled_win_rate  # NaN


def test_benjamini_hochberg_pass_all_significant():
    pvals = {0.0: 0.001, 0.4: 0.002, 0.45: 0.003, 0.5: 0.004, 0.55: 0.0001}
    result = benjamini_hochberg_pass(pvals, alpha=0.05)
    assert all(result.values())


def test_benjamini_hochberg_pass_none_significant():
    pvals = {0.0: 0.9, 0.4: 0.8, 0.45: 0.7, 0.5: 0.6, 0.55: 0.5}
    result = benjamini_hochberg_pass(pvals, alpha=0.05)
    assert not any(result.values())


def test_benjamini_hochberg_pass_step_up_rule_rescues_smaller_p_values():
    # Classic BH property: a moderately-sized p-value can still survive if enough
    # SMALLER p-values are also present, even though it wouldn't survive alone
    # against alpha/m.
    pvals = {0.0: 0.01, 0.4: 0.02, 0.45: 0.03, 0.5: 0.04, 0.55: 0.05}
    result = benjamini_hochberg_pass(pvals, alpha=0.05)
    assert all(result.values())


def test_evaluate_screening_verdict_rejects_significant_but_negative_pnl():
    # Encodes the AUD/USD lesson directly: a threshold must NOT pass just because
    # its p-value is small if the pooled net P&L is actually negative.
    results = {
        0.50: PooledThresholdResult(
            min_confidence=0.50, total_trades=1000, pooled_win_rate=0.30,
            win_rate_p_value=0.001, total_net_pnl_pct=-50.0,
            per_trade_t_p_value=0.001, per_trade_wilcoxon_p_value=0.001,
            per_fold_win_rates=[0.3, 0.3, 0.3, 0.3, 0.3],
        ),
    }
    verdict = evaluate_screening_verdict(results)
    assert not verdict.passed


def test_evaluate_screening_verdict_accepts_payoff_asymmetry_shape():
    # Encodes the USD/CHF ratio-variant lesson: below-50% win rate (not
    # win-rate-significant) but a significant, positive per-trade mean should
    # still pass via the per-trade path.
    results = {
        0.50: PooledThresholdResult(
            min_confidence=0.50, total_trades=1000, pooled_win_rate=0.48,
            win_rate_p_value=0.90, total_net_pnl_pct=40.0,
            per_trade_t_p_value=0.001, per_trade_wilcoxon_p_value=0.001,
            per_fold_win_rates=[0.48] * 5,
        ),
    }
    verdict = evaluate_screening_verdict(results)
    assert verdict.passed
    assert verdict.passing_thresholds == [0.50]


def test_evaluate_screening_verdict_passes_with_clean_positive_result():
    results = {
        thr: PooledThresholdResult(
            min_confidence=thr, total_trades=1000, pooled_win_rate=0.52,
            win_rate_p_value=0.01, total_net_pnl_pct=50.0,
            per_trade_t_p_value=0.01, per_trade_wilcoxon_p_value=0.01,
            per_fold_win_rates=[0.52] * 5,
        )
        for thr in [0.0, 0.40, 0.45, 0.50, 0.55]
    }
    verdict = evaluate_screening_verdict(results)
    assert verdict.passed
    assert verdict.passing_thresholds == [0.0, 0.40, 0.45, 0.50, 0.55]
