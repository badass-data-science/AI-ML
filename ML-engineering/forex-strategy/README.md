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
- **forex-etl** — reused for `granularity_to_seconds_map` (converting a trade's
  `test_exit_bar_offset` bars into a real exit timestamp for rollover-crossing math),
  the same canonical map forex-ML itself uses rather than a local duplicate.
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
and can filter on `column_y` — always `"triple_barrier"` for a current training run,
but filtering on it still matters: it excludes older, pre-migration model versions
(registered before forex-ML switched to triple-barrier labeling, still tagged
`pd_lead`/`volatility_lead` or untagged entirely) from the search:

```python
from mlflow.tracking import MlflowClient
from forex_strategy.model_registry import find_model_version, load_keras_model, load_test_predictions

client = MlflowClient(tracking_uri="sqlite:///../forex-ML/mlflow.db")
resolved = find_model_version(client, "forex-lstm", "EUR/USD", "H1", column_y="triple_barrier")
model = load_keras_model(resolved)
predictions = load_test_predictions(client, resolved.run_id, "/tmp/downloaded")
```

`load_test_predictions` downloads and loads forex-ML's `<run_uid>_predictions.npz`
artifact — `lstm_pred_proba`, `test_timestamp`/`test_price`/`test_spread`/`test_y_raw`/
`test_exit_bar_offset`/`test_realized_volatility`, plus the existing correctness
booleans.

## Running the backtest

```bash
uv run python -m forex_strategy.run_backtest \
    --tracking-uri sqlite:///../forex-ML/mlflow.db --instrument EUR/USD --granularity H1 \
    --use-live-swap-cost --flatten-before-rollover --size-by-realized-volatility
```

Each test row's predicted class maps to a position (highest class → long, lowest
class → short, middle → flat, via `predicted_classes_to_positions` — forex-ML's
triple-barrier label mapping puts a profit-take hit in the highest class and a
stop-loss hit in the lowest, so this is a direct read of the label, not a tercile
threshold), and `simulate_trades` computes P&L net of cost:

- **Spread** — charged as the full round-trip cost, always.
- **Swap/rollover** (`--long-swap-cost-pct-per-night`/`--short-swap-cost-pct-per-night`,
  or `--use-live-swap-cost` to fetch real current rates instead — see
  forex-ML's `forex_ml/data/swap_rates.py`) — charged once per 5pm New York
  rollover boundary *actually crossed* between entry and exit (via forex-ML's
  `count_rollovers_crossed`, DST-aware), not once per bar held. Direction-aware:
  a long position is charged the long rate, a short position the short rate —
  these are independently-signed real OANDA fields (e.g. one side can be a net
  credit while the other is a cost), not one rate mirrored with a flip. Exit
  timestamp is computed from each row's own `test_exit_bar_offset` (how many bars
  the triple-barrier label actually took to resolve), not a fixed `lookahead` — a
  real, variable holding period per trade.
- **The 5pm-NY flatten rule** (`--flatten-before-rollover`) — instead of paying swap,
  any trade whose holding period would cross a rollover is skipped entirely
  (`BacktestResult.n_flattened_for_rollover` reports how many).
- **Realized-volatility-gated position sizing** (`--size-by-realized-volatility`,
  `--target-volatility`, `--max-position-size`) — scales each trade's size by
  `target_volatility / test_realized_volatility` (clipped to `[0, max_position_size]`),
  the standard volatility-targeting idea (size inversely with risk) applied to
  forex-ML's `realized_volatility` passthrough column — a fixed 12-bar backward-looking
  average of already-observed per-bar volatility, read straight off the same model's
  own predictions artifact. No second model to train, register, or keep row-aligned
  by timestamp, unlike the old `volatility_lead`-model approach this replaced.

This makes sense against forex-ML's current production target, triple-barrier
labeling, which `find_model_version(..., column_y="triple_barrier")` filters for —
this also excludes older, pre-migration model versions (still tagged `pd_lead`/
`volatility_lead`, from before forex-ML's switch away from fixed-horizon labeling)
from being picked up as "the latest registered version for this pair" regardless of
scheme.

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

Of forex-etl's three newest sources, swap rates are now consumed for real (see
below); economic calendar and positioning data are still just ingested and
available for future work (e.g. a calendar-aware volatility overlay, a
positioning-based contrarian signal), with nothing here reading them yet.

**Post-roadmap update:** after all 9 phases above were done, forex-ML made the
production switch phase 5 anticipated but deferred — triple-barrier labeling now
*replaces* `pd_lead`/`volatility_lead` as the only trainable target, rather than
sitting alongside them as a standalone utility. This package was updated to match:
`column_y="triple_barrier"` everywhere phase 2/6 said `pd_lead`/`volatility_lead`,
the second-model volatility lookup from phase 6 is gone (replaced by the
`realized_volatility` passthrough column — see "Running the backtest" above), and
exit-timestamp math uses each row's real `test_exit_bar_offset` instead of a fixed
`lookahead`.

**Second post-roadmap update:** phase 4's swap-rate ingestion, similarly deferred
at the time, is now wired in for real too — `--use-live-swap-cost` fetches actual
OANDA long/short financing rates (via `forex_ml.data.swap_rates`) instead of the
manual `--long-swap-cost-pct-per-night`/`--short-swap-cost-pct-per-night` constants,
which now serve only as the fallback when no live snapshot exists yet. This also
fixed a real, previously-latent bug in the shared `InfluxDbTool` helper (both
repos depend on): its `unix_epoch_s` conversion assumed nanosecond-precision
timestamps and silently produced values 1000x too small for a certain class of
Flux query shape. It never affected this package (nothing here reads InfluxDB
directly), but the fix landed in the same change since the new swap-rate read
path would have hit it.

## Tests

```bash
uv run pytest -v
```

`test_backtest.py` is pure unit tests against synthetic arrays. `test_model_registry.py`
and `test_run_backtest.py` each have real end-to-end tests: `conftest.py`'s
`trained_triple_barrier_model` fixture trains and registers a real (tiny) model into a
scratch MLflow store via forex-ML's own `train_and_evaluate`, so this package's
model-loading and backtest code run against the real MLflow API and real logged
artifacts, not hand-mocked stand-ins — including a regression test proving the
`column_y` tag filter, not just "latest version," is what excludes a pre-migration
`pd_lead`-tagged version registered for the same pair.
