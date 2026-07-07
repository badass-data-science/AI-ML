"""Cost-aware backtest: turn a model's per-row class predictions into a trade
decision, and simulate P&L net of spread cost.

Class semantics match forex_ml.data.splitting.TimeSeriesSplitter._compute_outcome:
class 0 = lowest tercile of column_y, class 1 = middle, class 2 = highest tercile.
For a DIRECTIONAL target like pd_lead (% price change), that maps naturally onto
short / no-trade / long. This backtest assumes a directional target -- running it
against a magnitude-only target (volatility_lead, forex-ML's current default) isn't
meaningful on its own, since "high volatility" implies no direction. Volatility is
meant to feed position SIZING on top of a directional decision (see forex-ML's
README Roadmap, phase 6), not substitute for one. Callers should check the source
run's logged `column_y` param (see forex-ML's README) before treating `y_raw` as a
directional P&L input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BacktestResult:
    n_rows: int
    n_trades: int
    win_rate: float
    gross_pnl_pct: float
    cost_pct: float
    net_pnl_pct: float
    per_trade_net_pnl_pct: np.ndarray


def predicted_classes_to_positions(pred_proba: np.ndarray, min_confidence: float = 0.0) -> np.ndarray:
    """+1 (long) for the highest-tercile class, -1 (short) for the lowest-tercile
    class, 0 (flat) for the middle class. `min_confidence` is an optional hurdle on
    the winning class's own probability -- rows below it are forced flat, since a
    low-confidence call is exactly the kind of trade least likely to clear costs."""
    if pred_proba.ndim != 2 or pred_proba.shape[1] != 3:
        raise ValueError(f"Expected a (n_rows, 3) array of class probabilities, got shape {pred_proba.shape}")

    pred_idx = np.argmax(pred_proba, axis=1)
    confidence = np.max(pred_proba, axis=1)
    positions = np.where(pred_idx == 2, 1, np.where(pred_idx == 0, -1, 0))
    return np.where(confidence >= min_confidence, positions, 0)


def simulate_trades(
    positions: np.ndarray, pd_lead_pct: np.ndarray, spread: np.ndarray, price: np.ndarray,
) -> BacktestResult:
    """Spread-only cost model: no rollover/swap yet (forex-ML README Roadmap phases
    4/6 add that once swap rates are ingested). Cost is charged as the full
    round-trip spread -- buying at ask and later selling at bid (or the reverse, for
    a short) costs one full spread-width relative to the mid price `pd_lead_pct` is
    measured from. The entry bar's spread stands in for both legs since the exit
    bar's spread isn't available here (forex-ML computes `spread_close_lead` in
    Stage 1 but doesn't currently plumb it through Splits -- a natural refinement
    once that's wired up).

    `pd_lead_pct` is forex_ml's pd_lead target: 100 * (exit_price - entry_price) /
    entry_price, i.e. already a percentage -- see this module's docstring for why
    that's the only target this makes sense against. `spread`/`price` are the raw
    (non-percentage) price-unit values from Splits.test.
    """
    if not (len(positions) == len(pd_lead_pct) == len(spread) == len(price)):
        raise ValueError("positions/pd_lead_pct/spread/price must all be the same length")

    is_trade = positions != 0
    gross_pnl_pct = positions * pd_lead_pct
    cost_pct = np.where(is_trade, 100.0 * spread / price, 0.0)
    net_pnl_pct = np.where(is_trade, gross_pnl_pct - cost_pct, 0.0)

    n_trades = int(is_trade.sum())
    per_trade = net_pnl_pct[is_trade]

    return BacktestResult(
        n_rows=len(positions),
        n_trades=n_trades,
        win_rate=float((per_trade > 0).mean()) if n_trades else 0.0,
        gross_pnl_pct=float(gross_pnl_pct[is_trade].sum()) if n_trades else 0.0,
        cost_pct=float(cost_pct[is_trade].sum()) if n_trades else 0.0,
        net_pnl_pct=float(per_trade.sum()) if n_trades else 0.0,
        per_trade_net_pnl_pct=per_trade,
    )
