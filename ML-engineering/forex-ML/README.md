# forex-ML

LSTM training pipeline for Forex time series data pulled from InfluxDB. Three stages,
each independently orchestrated, versioned, and tracked:

```
InfluxDB ──▶ prepare (feature engineering) ──▶ split (windows/normalize) ──▶ train (LSTM + eval)
             forex_ml/data/features.py         forex_ml/data/splitting.py    forex_ml/training/
```

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
`train.mlflow_experiment_name`. Inspect runs with:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

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
