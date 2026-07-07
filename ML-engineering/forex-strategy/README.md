# forex-strategy

Trade selection, cost-aware backtesting, and execution logic for the models trained in the
sibling [`forex-ML`](../forex-ML) project.

`forex-ML` answers "what will this time series do next." This project answers a different
question: "given a prediction, is there an actual trade here that would make money after
spread and rollover/swap fees?" That's a distinct concern from model training — it needs
P&L simulation rather than accuracy metrics, doesn't need Spark or TensorFlow at its core,
and is the natural place for position sizing and (eventually) paper/live execution logic to
live. Keeping it a separate package keeps forex-ML focused on research and modeling.

## Stack

- **MLflow** — loads trained models and their metadata from forex-ML's tracking store,
  rather than retraining or duplicating anything.
- **pandas / numpy** — backtest simulation core.
- **pydantic** — config models, once there's configuration to validate.

## Setup

```bash
uv sync --extra dev
```

`forex-ml-training` (the forex-ML package) is a local editable path-dependency (see
`[tool.uv.sources]` in `pyproject.toml`), so this repo must be checked out as a sibling
directory of `forex-ML` for `uv sync` to resolve it.

## Roadmap

This project is being built incrementally, one phase per work session. Status as of this
writing:

1. ~~Scaffold `forex-strategy`~~ — **done**. Package skeleton only, no backtest logic yet.
2. **forex-ML: backtest-enabling plumbing** — persist test-set timestamps and raw price
   alongside the existing feature/label arrays; log raw predicted probabilities (not just
   correct/incorrect booleans) as an MLflow artifact; tag registered model versions by
   instrument/granularity/config so they can be looked up without grepping run params.
3. **MLflow model loading + core backtest/P&L simulation** — spread-only cost model first.
4. **forex-etl: swap/rollover rate ingestion** — new OANDA data source.
5. **forex-ML: cost-aware relabeling** — triple-barrier / net-of-cost target construction.
6. **Extend the backtest** — swap cost, volatility-gated position sizing, the 5pm-NY
   flatten rule.
7. **forex-etl: economic calendar ingestion.**
8. **forex-etl: OANDA positioning/order-book ingestion.**
9. **forex-ML: cross-pair feature-impact reuse** — no new ingestion required, all 7 major
   pairs are already collected.

## Tests

```bash
uv run pytest -v
```

Currently just a package-import smoke test — real test coverage arrives with phase 3's
backtest logic.
