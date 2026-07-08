# forex-strategy

Trade selection, cost-aware backtesting, and execution logic for the models trained in the
sibling [`forex-ML`](../forex-ML) project.

`forex-ML` answers "what will this time series do next." This project answers a different
question: "given a prediction, is there an actual trade here that would make money after
spread and rollover/swap fees?" That's a distinct concern from model training — it needs
P&L simulation rather than accuracy metrics, doesn't need Spark at its core, and is the
natural place for position sizing and (eventually) paper/live execution logic to live.
Keeping it a separate package keeps forex-ML focused on research and modeling.

## Stack

- **MLflow** — loads trained models and their metadata from forex-ML's tracking store
  (`forex_strategy/model_registry.py`), rather than retraining or duplicating anything.
- **TensorFlow/Keras** — needed transitively to actually load a forex-ML model
  (`mlflow.keras.load_model`); not used directly by this package's own code otherwise.
- **forex-etl** — reused for `granularity_to_seconds_map` (converting `lookahead` bars
  into a real exit timestamp for rollover-crossing math), the same canonical map
  forex-ML itself uses rather than a local duplicate.
- **pandas / numpy** — backtest simulation core (`forex_strategy/backtest.py`).

## Setup

```bash
uv sync --extra dev
```

`forex-ml-training` (the forex-ML package) and `forex-etl` are local editable
path-dependencies (see `[tool.uv.sources]` in `pyproject.toml`), so this repo must be
checked out as a sibling of both `forex-ML` and `Data-Science/Data-Engineering/ETL`
for `uv sync` to resolve them.

## Finding and loading a model

Every (instrument, granularity, config) trains under the same shared registered-model
name in forex-ML (`forex-lstm` by default) — see forex-ML's README, "Finding the model
for (instrument, granularity)". `forex_strategy/model_registry.py` wraps that lookup,
and can filter on `column_y` (`pd_lead` vs. `volatility_lead`) since two different
target models commonly share the same pair:

```python
from mlflow.tracking import MlflowClient
from forex_strategy.model_registry import find_model_version, load_keras_model, load_test_predictions

client = MlflowClient(tracking_uri="sqlite:///../forex-ML/mlflow.db")
resolved = find_model_version(client, "forex-lstm", "EUR/USD", "H1", column_y="pd_lead")
model = load_keras_model(resolved)
predictions = load_test_predictions(client, resolved.run_id, "/tmp/downloaded")
```

`load_test_predictions` downloads and loads forex-ML's `<run_uid>_predictions.npz`
artifact — `lstm_pred_proba`, `test_timestamp`/`test_price`/`test_spread`/`test_y_raw`,
plus the existing correctness booleans.

## Running the backtest

```bash
uv run python -m forex_strategy.run_backtest \
    --tracking-uri sqlite:///../forex-ML/mlflow.db --instrument EUR/USD --granularity H1 \
    --swap-cost-pct-per-night 0.02 --flatten-before-rollover --use-volatility-sizing
```

Each test row's predicted class maps to a position (highest tercile → long, lowest
tercile → short, middle → flat, via `predicted_classes_to_positions`), and
`simulate_trades` computes P&L net of cost:

- **Spread** — charged as the full round-trip cost, always.
- **Swap/rollover** (`--swap-cost-pct-per-night`) — charged once per 5pm New York
  rollover boundary *actually crossed* between entry and exit (via forex-ML's
  `count_rollovers_crossed`, DST-aware), not once per bar held.
- **The 5pm-NY flatten rule** (`--flatten-before-rollover`) — instead of paying swap,
  any trade whose holding period would cross a rollover is skipped entirely
  (`BacktestResult.n_flattened_for_rollover` reports how many).
- **Volatility-gated position sizing** (`--use-volatility-sizing`) — looks up a
  *second*, `volatility_lead`-trained registered version for the same pair, and scales
  each trade's size down as that model's predicted volatility tercile rises
  (`position_size_from_predicted_volatility_class`), the standard volatility-targeting
  idea (size inversely with risk) applied to forex-ML's ordinal 3-class prediction
  rather than a fabricated continuous magnitude. The two models' test sets must be
  row-aligned by timestamp (same `n_back`/`lookahead`/split configuration) — checked
  explicitly, not assumed.

This only makes sense against a **directional** target. forex-ML's 3-class scheme
(lowest/middle/highest tercile of `column_y`) maps naturally onto short/flat/long for
`pd_lead` (a % price change), but `volatility_lead` is a magnitude with no direction —
`find_model_version` is called with `column_y="pd_lead"` so it can only ever resolve a
directional model in the first place (this also fixes a latent gap: without the
`column_y` filter, "the latest registered version for this pair" could easily resolve
to whichever target was trained *most recently*, regardless of which one the caller
actually wants).

## Roadmap

This project was built incrementally, one phase per work session. All 9 planned phases
are now done:

1. ~~Scaffold `forex-strategy`~~ — **done**. Package skeleton only, no backtest logic yet.
2. ~~forex-ML: backtest-enabling plumbing~~ — **done**. Test-set timestamp/price/spread/
   raw target value persisted alongside the existing feature/label arrays; raw predicted
   probabilities logged as an MLflow artifact; registered model versions tagged by
   instrument/granularity/config-signature/`column_y` so runs on different targets never
   collapse together in `multiple_comparisons`, and so two target models for the same
   pair can be told apart.
3. ~~MLflow model loading + core backtest/P&L simulation~~ — **done**. Spread-only cost
   model, directional-target guard, all exercised end-to-end in tests against a real
   tiny model trained and registered into a scratch MLflow store (not mocked).
4. ~~forex-etl: swap/rollover rate ingestion~~ — **done**. `SwapRateETL`/`SwapRateRecord`,
   a new `swap-rate` InfluxDB measurement, scheduled ~15 minutes before the 5pm NY
   rollover cutoff.
5. ~~forex-ML: cost-aware relabeling~~ — **done**. `triple_barrier_labels` (Lopez de
   Prado's method), cost-aware (spread + swap, DST-aware rollover counting) — a
   standalone labeling utility, not yet wired into Stage 1's production `column_y`.
6. ~~Extend the backtest~~ — **done**. Swap cost, the 5pm-NY flatten rule, and
   volatility-gated position sizing (a second registered model, combined by timestamp
   alignment) — see "Running the backtest" above.
7. ~~forex-etl: economic calendar ingestion~~ — **done**. `EconomicCalendarETL`
   (Finnhub, not Oanda — a separate provider/credential), a new
   `economic-calendar-event` InfluxDB measurement.
8. ~~forex-etl: OANDA positioning/order-book ingestion~~ — **done**. `PositioningETL`
   (back to Oanda's own API/token), a new `positioning-bucket` measurement — raw
   per-price-bucket data, not a collapsed "overall % long/short" stat Oanda's exact
   normalization wasn't confirmable here.
9. ~~forex-ML: cross-pair feature-impact reuse~~ — **done**.
   `analyze_cross_pair_feature_impact`/`--cross-pair-candidates` — every existing
   screening technique now also works with candidates drawn from a different
   instrument than the target, no new ingestion required.

None of forex-etl's three newest sources (swap rates, economic calendar, positioning)
are consumed by this package's backtest yet beyond the swap-cost/flatten-rule wiring
already in place — the calendar and positioning data are ingested and available for
future work (e.g. a calendar-aware volatility overlay, a positioning-based contrarian
signal), but nothing here reads them yet.

## Tests

```bash
uv run pytest -v
```

`test_backtest.py` is pure unit tests against synthetic arrays. `test_model_registry.py`
and `test_run_backtest.py` each have real end-to-end tests: `conftest.py`'s
`trained_pd_lead_model`/`trained_volatility_lead_model`/
`trained_pd_lead_and_volatility_models` fixtures train and register real (tiny) models
into a scratch MLflow store via forex-ML's own `train_and_evaluate`, so this package's
model-loading and backtest code run against the real MLflow API and real logged
artifacts, not hand-mocked stand-ins — including a regression test proving the
`column_y` tag filter, not just "latest version," is what picks the right model when
both a `pd_lead` and a `volatility_lead` version are registered for the same pair.
