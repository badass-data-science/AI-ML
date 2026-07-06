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
secrets directly. `forex_ml/data/influx_source.py` accesses it as
`database_config.INFLUXDB_URL` (module attribute, resolved fresh on each call) rather
than `from database_config import INFLUXDB_URL` — the latter would freeze the
resolved secret the moment ANYTHING imports the module (including just pytest
collecting an unrelated test file), permanently, for the life of the process, with
no way for a later test to substitute different credentials. See
`tests/test_secrets_isolation.py` for the regression test and the real bug this
guards against.

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

Also reports an effect size for both ACF and PACF, not just the significance cutoff — the same statistical-vs-practical-significance gap as the ADF/KPSS diagnostics below. With the hundreds-to-thousands of bars typical here, the confidence interval narrows enough that a tiny correlation (0.02) can still be "significant," which would make `suggested_min_lookback` track sample size rather than real memory. Reported per series:

- `acf_magnitude_at_suggested_lookback` / `pacf_magnitude_at_suggested_lookback` — the raw |ACF|/|PACF| value right at the statistically-suggested cutoff, and `*_max_abs_magnitude` — the largest magnitude across all computed lags. Together these show how small "still significant" actually is.
- `practical_min_lookback` / `practical_min_lookback_pacf` — the first lag where |ACF|/|PACF| drops below a fixed threshold (`--practical-threshold`, default 0.1) that does **not** shrink as the sample grows, giving a second answer anchored to correlation strength rather than significance.

PACF gets its own `suggested_min_lookback_pacf`/`practical_min_lookback_pacf` rather than reusing ACF's, since PACF is the more standard tool for spotting an AR cutoff (it tends to drop sharply at the true order, where ACF decays gradually) — the two can legitimately disagree, and both are printed.

```bash
uv run python -m forex_ml.diagnostics.stationarity --instrument EUR/USD --granularity H1
```

Runs ADF and KPSS together on each `split.columns_x` column (`forex_ml/diagnostics/stationarity.py`) — they test opposite null hypotheses (ADF: "has a unit root"; KPSS: "is stationary"), so using both catches blind spots either test misses alone. Both agreeing is a strong signal; disagreeing gets reported as `inconclusive` rather than silently picking one test's answer.

Also reports an effect size for each test, not just the two p-values — matched to
what each test actually estimates rather than one generic number:

- **ADF** — `phi_hat` (the AR(1) coefficient) and its `half_life_bars`, extracted
  directly from the ADF regression's own fitted model rather than a second
  regression. With the hundreds-to-thousands of bars typical here, ADF/KPSS will
  tend to reject the unit-root null for almost any realistic series, even a highly
  persistent one (`phi_hat` near 1, behaving practically like a random walk over the
  horizons that matter for `n_back`/`lookahead`) — statistical significance isn't
  the same as practical significance, and large samples widen that gap. A short
  half-life and a long half-life can both get called "stationary"; only `phi_hat`/
  `half_life_bars` tells them apart.
- **KPSS** — `kpss_ratio_to_5pct`, the raw LM statistic (already computed internally
  by `statsmodels.tsa.stattools.kpss` and otherwise discarded, keeping only the
  p-value) as a multiple of its own 5% critical value. KPSS doesn't have as clean a
  real-units effect size as ADF's half-life — its statistic doesn't decompose into
  bars or any other physical unit — but the raw magnitude still carries graduated
  information a bare p-value throws away: a ratio of 3.0 is a much stronger
  non-stationarity signal than 1.01, even though both cross the "significant" line
  identically.

```bash
uv run python -m forex_ml.evaluation.multiple_comparisons --experiment forex-lstm
```

With 14 `(instrument, granularity)` pairs each getting their own "does the LSTM beat
the baseline?" test, some pair will look significant by chance even if none has real
signal. `forex_ml/evaluation/multiple_comparisons.py` runs McNemar's test per (pair,
model configuration) (the correct paired test for two classifiers evaluated on the
same test rows — every training run now saves a `predictions.npz` artifact with
per-row correctness for exactly this) and applies a Benjamini-Hochberg FDR correction
across all of them, so the reported significant count reflects the whole comparison,
not each result judged against a raw, uncorrected alpha.

This is designed for building up data **incrementally** — with one local GPU and no
cloud compute, pairs get trained individually over time rather than all 14 at once,
and architecture search (trying different layer counts/widths, activation functions,
epochs, etc. on the same pair) is expected to be a frequent, ongoing part of that.
That workflow has two genuinely different reasons to retrain the same pair, and this
module treats them differently on purpose:

- **Same configuration, more data accumulated** — only the most recent run counts.
  `report_across_pairs` groups by `(instrument, granularity, _model_config_signature)`
  — a hash of every logged `TrainParams` field, not just layer count/width — and
  pulls runs ordered by `start_time` descending, skipping older runs once a
  (pair, configuration)'s latest one is found. This is genuinely one hypothesis
  re-evaluated with an updated estimate, not a new one.
- **Different configuration on the same pair (architecture search)** — treated as a
  *separate* hypothesis with its own entry in the report and its own slot in the BH
  correction. Collapsing this to "whichever configuration was trained most recently"
  would silently discard every other configuration's result from the correction,
  understating how many hypotheses were actually tested — exactly the kind of
  researcher-degrees-of-freedom problem this module exists to catch, and one that
  gets worse the more often architecture search happens, not better.

Because the correction is a function of however many (pair, configuration)
combinations currently have data, it re-tightens every time a new one is added — a
result marked significant today can stop being significant once more are added,
purely because there are more hypotheses to correct for, not because that result's
own p-value changed. Re-run this after each new training run rather than treating an
early verdict as final.

```bash
uv run python -m forex_ml.evaluation.rolling_cv --instrument EUR/USD --granularity H1 \
  --n-folds 5 --window sliding --min-train-bars 2000 --val-bars 500 --test-bars 500
```

A single train/val/test split (however it was chosen) is one sample from one slice of
history — a good or bad result could just be that slice, not the configuration.
`forex_ml/evaluation/rolling_cv.py` walks a train/val/test window forward through the
timeline, retraining fresh each fold, and reports the *distribution* of test results
(LSTM and both baselines) across folds rather than a single number — mean/std/min/max,
and whether the LSTM clears the majority baseline in every fold or just on average.

Two window types, chosen via `--window`:

- **`sliding`** — the training block has a fixed length (`--min-train-bars`) and
  slides forward with the fold, so every fold trains on comparably-recent history.
  More robust to regime change (older data ages out), at the cost of using less data
  per fold than is actually available by the final fold.
- **`expanding`** — the training block always starts at the first bar and grows by
  one test-block's worth (`--test-bars`) each fold. Uses all available data, at the
  cost of assuming older data is still as relevant as recent data — a stronger
  stationarity assumption.

Every fold's boundaries are purge-gap aware, same as the single split: `--purge-bars`
defaults to `max(feature.n_back, feature.lookahead)` from `params.yaml` (overridable),
purging that many bars on both sides of each fold's train/val and val/test boundary so
no fold's window or label overlaps an adjacent split — see the purge-gap note under
Configuration above for why.

This is a **robustness diagnostic only** — it doesn't change what gets deployed. Each
fold trains and logs to its own MLflow experiment (`<experiment>-rolling-cv`), tagged
with its fold index and window type, and is never registered in the model registry.
Keeping it in a separate experiment matters: `multiple_comparisons.py` scans one
experiment for one "official" run per `(pair, configuration)`, and rolling-CV fold
runs (same configuration, deliberately re-trained many times across time windows)
would otherwise either get silently collapsed into that pool as bogus "retrains" or
inflate it with runs that were never meant to be independent hypotheses.

Two related but heavier extensions are possible **future next steps**, not built
here:

- **Model/architecture selection tool** — using performance averaged across folds to
  choose between competing configurations before committing to a final single
  train/val/test run, rather than just reporting how stable *one* configuration is.
  This would need to feed fold results into a selection decision, and interacts with
  the multiple-comparisons machinery above (comparing many configurations' fold
  averages is itself another layer of multiple comparisons).
- **Walk-forward retraining strategy** — turning this into the actual production
  retraining cadence, where each fold's model is a real deployment candidate for its
  period rather than a diagnostic artifact. Bigger change: touches the model
  registry, deployment selection, and is meaningfully heavier compute on a single
  local GPU than a one-off diagnostic run.

## Tests

```bash
uv run pytest -v -m "not integration"   # fast unit suite, no Docker needed
uv run pytest -v -m integration         # real InfluxDB integration tests (needs Docker)
```

The unit suite covers feature engineering (Spark, against synthetic candles), the
time-based split/normalize/discretize logic including rolling-fold boundary math
(pure pandas, no Spark needed), model construction, and one true end-to-end smoke
test: synthetic tensors → 1-epoch fit using the *real* validation split → held-out
test evaluation → MLflow run assertion. That last test is what would have caught the
original bug where the precomputed validation set was silently discarded in favor of
`validation_split=`. It also covers every evaluation/diagnostics module — baselines,
class balance, ACF/PACF, ADF/KPSS stationarity, multiple-comparisons BH-FDR
correction, and rolling CV — each with at least one real end-to-end test (real
Spark-engineered Stage-1 output and/or a real local MLflow store), not just
unit tests against hand-built arrays.

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
