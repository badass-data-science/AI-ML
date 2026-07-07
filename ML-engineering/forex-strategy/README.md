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
- **pandas / numpy** — backtest simulation core (`forex_strategy/backtest.py`).

## Setup

```bash
uv sync --extra dev
```

`forex-ml-training` (the forex-ML package) is a local editable path-dependency (see
`[tool.uv.sources]` in `pyproject.toml`), so this repo must be checked out as a sibling
directory of `forex-ML` for `uv sync` to resolve it.

## Finding and loading a model

Every (instrument, granularity, config) trains under the same shared registered-model
name in forex-ML (`forex-lstm` by default) — see forex-ML's README, "Finding the model
for (instrument, granularity)". `forex_strategy/model_registry.py` wraps that lookup:

```python
from mlflow.tracking import MlflowClient
from forex_strategy.model_registry import find_model_version, load_keras_model, load_test_predictions

client = MlflowClient(tracking_uri="sqlite:///../forex-ML/mlflow.db")
resolved = find_model_version(client, "forex-lstm", "EUR/USD", "H1")
model = load_keras_model(resolved)
predictions = load_test_predictions(client, resolved.run_id, "/tmp/downloaded")
```

`load_test_predictions` downloads and loads forex-ML's `<run_uid>_predictions.npz`
artifact — `lstm_pred_proba`, `test_timestamp`/`test_price`/`test_spread`/`test_y_raw`,
plus the existing correctness booleans.

## Running the backtest

```bash
uv run python -m forex_strategy.run_backtest \
    --tracking-uri sqlite:///../forex-ML/mlflow.db --instrument EUR/USD --granularity H1
```

`forex_strategy/backtest.py` is a **spread-only cost model**: each test row's predicted
class maps to a position (highest tercile → long, lowest tercile → short, middle →
flat, via `predicted_classes_to_positions`), and `simulate_trades` charges the full
round-trip spread against the realized `pd_lead` move. No rollover/swap yet — that
arrives in phase 6, once phase 4 ingests swap rates.

This only makes sense against a **directional** target. forex-ML's 3-class scheme
(lowest/middle/highest tercile of `column_y`) maps naturally onto short/flat/long for
`pd_lead` (a % price change), but `volatility_lead` (forex-ML's current default
target) is a magnitude with no direction — `run_backtest.backtest_from_mlflow` checks
the source run's logged `column_y` param and refuses to run against anything other
than `pd_lead`, rather than silently producing meaningless P&L numbers. Volatility is
meant to feed position *sizing* on top of a directional decision (phase 6), not
substitute for one.

## Roadmap

This project is being built incrementally, one phase per work session. Status as of this
writing:

1. ~~Scaffold `forex-strategy`~~ — **done**. Package skeleton only, no backtest logic yet.
2. ~~forex-ML: backtest-enabling plumbing~~ — **done**. Test-set timestamp/price/spread/
   raw target value persisted alongside the existing feature/label arrays; raw predicted
   probabilities logged as an MLflow artifact; registered model versions tagged by
   instrument/granularity/config-signature; `column_y` logged so runs on different
   targets never collapse together in `multiple_comparisons`.
3. ~~MLflow model loading + core backtest/P&L simulation~~ — **done**. Spread-only cost
   model, directional-target guard, all exercised end-to-end in tests against a real
   tiny model trained and registered into a scratch MLflow store (not mocked).
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

`test_backtest.py` is pure unit tests against synthetic arrays. `test_model_registry.py`
and `test_run_backtest.py` each have at least one real end-to-end test: `conftest.py`'s
`trained_pd_lead_model`/`trained_volatility_lead_model` fixtures train and register a
real (tiny) model into a scratch MLflow store via forex-ML's own `train_and_evaluate`,
so this package's model-loading and backtest code run against the real MLflow API and a
real logged artifact, not a hand-mocked stand-in.
