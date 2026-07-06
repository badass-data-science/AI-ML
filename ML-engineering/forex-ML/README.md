# forex-ML

LSTM training pipeline for Forex time series data pulled from InfluxDB. Three stages,
each independently orchestrated, versioned, and tracked:

```
InfluxDB ──▶ prepare (feature engineering) ──▶ split (windows/normalize) ──▶ train (LSTM + eval)
             forex_ml/data/features.py         forex_ml/data/splitting.py    forex_ml/training/
```

## Forward-fill contamination

Gaps in the raw candle data get forward-filled (weekends, holidays, thin liquidity)
before this pipeline ever sees it — a forward-filled bar has zero return, zero
volatility, and a stale spread by construction, which a model can't otherwise
distinguish from a genuinely quiet real market. `forex.etl.pipelines.ForwardFillInator`
(in the sibling `forex-etl` repo) now tracks which rows were imputed and writes an
`is_forward_filled` field alongside every bar in the `forward-filled candlestick`
InfluxDB measurement. `prepare_data_flow.py` carries it through as an available
feature column when present (and is a no-op when it isn't, for historical data
written before this field existed) — it's not in `split.columns_x` by default, so
opt in via `params.yaml` if you want the model to see it.

## Trading session features

`add_session_features` (`forex_ml/data/features.py`) adds `is_tokyo_session`,
`is_london_session`, `is_new_york_session`, and `is_london_new_york_overlap` (the
historically most liquid/volatile window) — a well-documented FX volatility driver
distinct from the existing day/week cyclical features, which capture broad calendar
seasonality but not which major market is currently open. Session boundaries are
fixed approximate UTC hours (not DST-aware), computed by plain arithmetic on the
epoch second rather than `F.hour(F.from_unixtime(...))` — the latter is silently
sensitive to Spark's SQL session timezone, which isn't guaranteed to be UTC. Like
`is_forward_filled`, these aren't in `split.columns_x` by default — opt in via
`params.yaml`.

## Stack

- **Prefect** — orchestration (`forex_ml/flows/`), matching the conventions already
  used by the ETL side (`Data-Science/Data-Engineering/ETL`)
- **MLflow** — experiment tracking + model registry (local SQLite backend by default)
- **DVC** — data/model-artifact versioning (`dvc.yaml`, keyed on `params.yaml`)
- **pydantic** — validated pipeline config (`forex_ml/config.py`, loaded from
  `params.yaml`)
- **pytest** — unit + end-to-end smoke tests (`tests/`)

## Setup

```bash
uv sync --extra dev
```

This installs pinned dependencies (PySpark, TensorFlow/Keras, MLflow, Prefect, DVC,
pydantic) plus editable installs of the sibling `forex-etl` and
`python-tools-and-shortcuts` packages — no `sys.path.append` hacks, no hardcoded
absolute paths. InfluxDB credentials are handled entirely by
`forex.etl.config.database_config` (AWS Secrets Manager-backed); nothing here touches
secrets directly.

## Configuration

Everything the pipeline needs — instrument/granularity lists, feature-engineering
params, split proportions, model hyperparameters — lives in `params.yaml`, validated
by `forex_ml/config.py` at load time. In particular, `split.columns_x` is checked
against the columns Stage 1 actually produces, so a config that references a feature
that doesn't exist fails immediately instead of three stages later.

Train/val/test are split strictly by timestamp (`TimeSeriesSplitter` in
`forex_ml/data/splitting.py`) — never shuffled — so training data is always
chronologically before validation, which is always before test. `split_flow.py` also
purges `max(n_back, lookahead)` bars on both sides of each split boundary: a window
reaches `n_back` bars backward and a label reaches `lookahead` bars forward, so
without a purge gap the row right at a boundary can have a window or label that
overlaps the adjacent split. That's not leakage in the sense of the model seeing
future inputs at inference time, but the two adjacent rows are highly autocorrelated,
which can optimistically bias the validation/test metric right at the seam (see
Lopez de Prado's *purged k-fold CV* for the general technique).

## Running a single pair

```bash
uv run python -m forex_ml.flows.prepare_data_flow --instrument EUR/USD --granularity H1
uv run python -m forex_ml.flows.split_flow        --instrument EUR/USD --granularity H1
uv run python -m forex_ml.flows.train_flow        --instrument EUR/USD --granularity H1
```

Every output path is keyed on `(instrument, granularity, n_back, lookahead)` (see
`forex_ml/paths.py`), so preparing multiple pairs never overwrites another pair's data
— the original notebooks wrote Stage 2's output to a single shared
`output/data.pickled` regardless of pair, which meant only one pair could be staged
for training at a time.

## Running everything via DVC

```bash
uv run dvc repro
```

`dvc.yaml` defines a `prepare → split → train` chain per `(instrument, granularity)`
pair listed in `params.yaml`'s `pairs:` (one independent chain per pair — see `dvc dag`
for the full graph). Re-running only re-executes stages whose deps or params actually
changed. A local DVC remote is configured as a placeholder
(`/home/emily/dvc-storage/forex-ml`) — swap it for S3/GCS via `dvc remote add` for
real shared storage.

## Scheduled retraining (optional)

```bash
uv run python -m forex_ml.flows.serve
```

Starts a weekly Prefect deployment that runs prepare → split → train for every pair in
`params.yaml`, mirroring `forex/flows/serve.py` on the ETL side. Most day-to-day use is
just the individual flows or `dvc repro` above; this is for unattended periodic
retraining.

## Experiment tracking

Every training run logs params, per-epoch train/val metrics, and a held-out **test**
evaluation to MLflow (`sqlite:///mlflow.db` by default — see `train.mlflow_tracking_uri`
in `params.yaml`), and registers the model in the MLflow Model Registry under
`train.mlflow_experiment_name`. Alongside the LSTM's own test metrics, every run also
logs two baselines from `forex_ml/evaluation/baselines.py` — `baseline_majority_test_accuracy`
(always predict the training set's most common class) and
`baseline_persistence_test_accuracy` (predict this period repeats the previous
period's actual class) — so `test_accuracy` is never read in isolation. Every run
also logs each split's actual class balance (`train_class_0_balance`,
`val_class_1_balance`, etc., from `forex_ml/evaluation/class_balance.py`) — train is
close to even by construction (thresholds come from train quantiles), but val/test
aren't guaranteed to be if the volatility regime has shifted between periods, and
this is the cheapest way to see that drift instead of it hiding inside a single
accuracy number.

This does NOT introduce lookahead into the pipeline: `class_balance()` only reads the
already-materialized `y` arrays and writes a metric — it never touches `X`, never
feeds back into the (already train-only) threshold computation in
`TimeSeriesSplitter`, and is never passed to `model.fit()`/`model.evaluate()` or
either callback. The test set enters the model exactly once, at the single
`model.evaluate(splits.test...)` call after training is finished, regardless of this
logging. There is a real but different risk worth naming: because this sits next to
`test_accuracy` in the same MLflow run, it puts test-set characteristics in front of
you at the same time as test performance — if you see "test period skewed toward
class 2" and go adjust thresholds/features/hyperparameters and retry, *that's* test-set
leakage, just introduced by the human in the loop rather than by the code. Treat it as
a diagnostic to explain a result you already have, not a signal to iterate against.

Inspect runs with:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Diagnostics

```bash
uv run python -m forex_ml.diagnostics.autocorrelation --instrument EUR/USD --granularity H1
```

Reports ACF/PACF-based sanity checks against a pair's real Stage-1 output (`forex_ml/diagnostics/autocorrelation.py`) — specifically, the first lag at which the target column's autocorrelation is no longer statistically distinguishable from zero. That's a floor on how much history carries *linear* signal, not a definitive answer (the LSTM can exploit nonlinear structure this can't see), but if `feature.n_back` in `params.yaml` is wildly larger than the suggested minimum, it's worth checking rather than assuming — `n_back=200`/`lookahead=4` were carried over from the original notebooks with no such check behind them.

```bash
uv run python -m forex_ml.diagnostics.stationarity --instrument EUR/USD --granularity H1
```

Runs ADF and KPSS together on each `split.columns_x` column (`forex_ml/diagnostics/stationarity.py`) — they test opposite null hypotheses (ADF: "has a unit root"; KPSS: "is stationary"), so using both catches blind spots either test misses alone. Both agreeing is a strong signal; disagreeing gets reported as `inconclusive` rather than silently picking one test's answer.

```bash
uv run python -m forex_ml.evaluation.multiple_comparisons --experiment forex-lstm
```

With 14 `(instrument, granularity)` pairs each getting their own "does the LSTM beat
the baseline?" test, some pair will look significant by chance even if none has real
signal. `forex_ml/evaluation/multiple_comparisons.py` runs McNemar's test per pair
(the correct paired test for two classifiers evaluated on the same test rows — every
training run now saves a `predictions.npz` artifact with per-row correctness for
exactly this) and applies a Benjamini-Hochberg FDR correction across all pairs, so
the reported significant count reflects the whole comparison, not each pair judged
against a raw, uncorrected alpha.

This is designed for building up data **incrementally** — with one local GPU and no
cloud compute, pairs get trained individually over time rather than all 14 at once.
Two things follow from that:

- If a pair gets retrained (new hyperparameters, more data, etc.), only its most
  recent run is used — `report_across_pairs` pulls runs ordered by `start_time`
  descending and skips older runs once a pair's latest one is found, so an earlier
  attempt never silently overwrites a later one depending on MLflow's internal
  ordering.
- The correction is a function of *how many pairs currently have data*, not the
  full set of 14. As more pairs get trained, the correction re-tightens across
  everything — a pair marked significant today can stop being significant once more
  pairs are added, purely because there are more tests to correct for, not because
  that pair's own result changed. Re-run this after each new pair rather than
  treating an early verdict as final; the CLI prints "N of M expected pairs have
  data so far" (reading the expected total from `params.yaml`) as a reminder.

## Tests

```bash
uv run pytest -v -m "not integration"   # fast unit suite, no Docker needed
uv run pytest -v -m integration         # real InfluxDB integration tests (needs Docker)
```

The unit suite covers feature engineering (Spark, against synthetic candles), the
time-based split/normalize/discretize logic (pure pandas, no Spark needed), model
construction, and one true end-to-end smoke test: synthetic tensors → 1-epoch fit
using the *real* validation split → held-out test evaluation → MLflow run assertion.
That last test is what would have caught the original bug where the precomputed
validation set was silently discarded in favor of `validation_split=`.

`tests/test_influx_integration.py` is a second tier: it spins up a real InfluxDB 2.x
container via Docker, seeds it with synthetic "forward-filled candlestick" rows using
the exact schema `forex_ml.data.influx_source` queries, and pulls it back through the
real Flux query + `InfluxDbTool` + pandas path — nothing on the DB boundary is mocked.
A second test in the same file runs the full Stage-1 flow (pull → engineer features →
Parquet) against that same container. It's excluded from the default run (needs
Docker, slower) and run separately in CI. Writing it honestly surfaced a real bug:
`prepare_data_flow`/`split_flow` used to call `spark.stop()` in a `finally` block,
which — since a JVM only ever has one active SparkContext — killed Spark out from
under any other flow or test fixture sharing the same process (e.g. `serve.py`'s
retrain loop calling all three flows back-to-back for every pair). Session lifecycle
is now the caller/process's responsibility, not the individual flow's.

CI (`.github/workflows/forex-ml-ci.yml`) runs lint (`ruff`), type-check (`mypy`), the
unit suite, and the Docker-backed integration suite on every push touching this
subtree.

## Legacy notebooks

`notebooks_legacy/` holds the three original notebooks this package replaced, kept for
ad-hoc exploration only — see `notebooks_legacy/README.md` for what replaced what.
