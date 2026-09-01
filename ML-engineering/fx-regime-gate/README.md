# fx-regime-gate

A two-signal situational-awareness dashboard: combines VIX level with
fx-pcn direction-flip burst status to answer one question — is a current
volatility spike actually showing up as structural change in FX pair
correlations, or is it (so far) an equity-only event?

This is a sketch/prototype living here for now. It reads fx-pcn's existing
output (does not run fx-pcn itself) and fetches VIX from yfinance.

## Why this exists

Built out of `../ideas.txt` idea #5 (direction-flip bursts as a
calendar-free event detector) and idea #2 (fx-pcn as a re-validation
trigger for forex-ML models), after testing showed:

- fx-pcn bursts don't detect FOMC events (rejected — the notebook's
  apparent Granger-causality signal there was a spurious periodicity
  artifact).
- fx-pcn bursts *do* correlate with elevated VIX, but only at the extreme
  tail (top ~6% of weeks), and VIX leads bursts by 1–10 weeks, not the
  reverse.

That makes the burst signal useless as a standalone predictor, but useful
as a **confirmation filter** on top of VIX — see the full methodology and
every validation test (shuffle placebo, reverse-direction placebo,
circular-shift placebo) in
`../../../../forex-partial-correlation-network/EDA/FINDINGS.md`.

## Usage

```
python3 gate.py
```

Requires `pandas`, `numpy`, `yfinance`, and fx-pcn's own output already
populated under `~/output/forex-partial-correlation-network/` (config:
`window-days-60---step-days-7---min-observations-30---max-lag-3---fdr-alpha-0.05---granularity-D`).

The script prints a status report including a "HOW TO READ THIS GATE"
section — read that before acting on the color, since this is a lagging
confirmation signal, not an early-warning one.

## Next steps (not done yet)

- No historical backtest of the gate's color transitions against anything
  actionable (e.g. did RED periods actually coincide with forex-ML model
  degradation) — that's the real test of whether idea #2 pays off.
- No persistence/scheduling; this is a point-in-time snapshot, run by hand.
- No visual/web dashboard — text report only.
