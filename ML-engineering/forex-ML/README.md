# forex-ML

LSTM training pipeline for Forex time series data pulled from InfluxDB. Three stages,
each independently orchestrated, versioned, and tracked:

```
InfluxDB ──▶ prepare (feature engineering) ──▶ split (windows/normalize) ──▶ train (LSTM + eval)
             forex_ml/data/features.py         forex_ml/data/splitting.py    forex_ml/training/
```

## Status (2026-07-14): forecasting-edge search closed on EUR/USD, no validated edge found

This project set out to answer one question: does this pipeline, applied to
major-pair forex data, produce a model whose predictions clear real trading
costs and hold up out of sample? After an extensive, well-instrumented search,
the honest answer on EUR/USD is **no** — not "not yet," but no validated edge
across everything tried, with a trustworthy measurement instrument confirming
it at every step.

**What was searched**, all against the same leak-proof, purge-gap-aware,
cost-aware pipeline (bidirectional triple-barrier labeling, rolling-window
out-of-sample validation, McNemar/multiple-comparisons-corrected significance
testing throughout):

- **2 model families**: an LSTM (which turned out to be seed-dependently
  unstable — see "Training instability" below — for reasons never fully
  resolved) and gradient-boosted trees (stable across every seed and pair
  tried, the more trustworthy of the two).
- **2 objectives**: 3-class direction classification, and net-return
  regression (predicting the actual expected P&L rather than just direction).
- **4 feature sets**: a 10-feature OHLCV/calendar/session baseline, plus three
  rounds of enrichment (multi-timeframe trend + volatility regime, momentum
  oscillator analogs, a cross-pair "USD strength" signal) — see "Richer
  features" below.
- **2 granularities**: H1 (the primary focus) and H4 (added specifically to
  test whether a coarser timescale had a better signal-to-noise ratio — see
  "Granularity comparison" below), plus lighter spot-checks on three more
  pairs at H1 confirming the same instability/stability pattern generalizes
  beyond EUR/USD.

**Every single "promising" single-window backtest result found along the
way — and there were several — failed to survive 5-fold, non-overlapping,
out-of-sample validation.** The cost structure itself was checked directly
against real data and found not to be the dominant obstacle (spread is only
~5% of a typical barrier-sized move; the breakeven win rate is a modest
~52.5%, not some unreachable bar). The actual limiting factor, confirmed
repeatedly: fold-to-fold and seed-to-seed noise (win rates ranging over 25+
percentage points depending on which window or seed happened to be tested)
dwarfs the few-percentage-point edge that would be needed to be profitable.

**The backtest itself was independently audited twice** before trusting any
of this — once turning up a real bug (wrong-direction trades priced using the
wrong side's outcome, fixed — see "Backtest fix" below), and once confirming
no further issues. The measurement instrument is trustworthy; what it
consistently measured is the absence of a validated edge in what was tried.

**Scope of this negative result — what it does NOT claim:** this is not a
claim that no exploitable edge exists in forex generally, or even in EUR/USD
at every timescale and feature set. Real avenues remain untried: order-flow/
L2 book data; genuine news/economic-calendar features (blocked so far by a
Finnhub API tier limitation, not a code issue); real retail-positioning data
(blocked by OANDA discontinuing the relevant endpoints); less liquid,
less-arbitraged pairs or asset classes entirely; a directly cost-aware
training objective rather than a proxy loss. This result closes out the
specific, wide-but-bounded search described above, not the whole space of
possible approaches.

**What this project actually produced**, independent of whether it found a
trading edge: a rigorous, reusable, leak-proof forex ML pipeline —
bidirectional cost-aware triple-barrier labeling, purge-gap-aware
train/val/test and rolling-window splitting, a full statistical-significance
and multiple-comparisons-correction toolkit, and an audited cost-aware
backtest engine (the sibling [`forex-strategy`](../forex-strategy)
project — see its own README and blog series for a from-scratch explanation
of how and why it works). That infrastructure is the real, portable asset
this investigation leaves behind, and is exactly what the next phase (a
different, less efficient market) would reuse rather than rebuild.

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

### GPU training

`pyproject.toml` depends on `tensorflow[and-cuda]`, which installs cuDNN/cuBLAS/etc.
as pip packages (`nvidia-*`) rather than requiring a system-wide CUDA toolkit. That's
enough for `tf.test.is_built_with_cuda()` to report `True`, but **not** enough for
TensorFlow to actually find and load those libraries at runtime — `uv`/pip install
them into the venv but don't add them to the dynamic linker's search path. Without
this, training silently runs on CPU with no error, just no GPU listed in
`tf.config.list_physical_devices("GPU")`.

Run this once per shell session before any training/GPU-dependent command (`train_flow`,
`rolling_cv`, or anything that imports `forex_ml.training.model`):

```bash
export LD_LIBRARY_PATH="$(find .venv/lib/python3.11/site-packages/nvidia -maxdepth 2 -type d -name lib | tr '\n' ':')${LD_LIBRARY_PATH}"
```

Verify it worked:

```bash
uv run python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Should print `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`, not
`[]`. This is a standard consequence of pip-installed (rather than system) CUDA on
Linux, not specific to this project's setup.

## Configuration

Everything the pipeline needs — instrument/granularity lists, feature-engineering
params, split proportions, model hyperparameters — lives in `params.yaml`, validated
by `forex_ml/config.py` at load time. In particular, `split.columns_x` is checked
against the columns Stage 1 actually produces, so a config that references a feature
that doesn't exist fails immediately instead of three stages later.

Train/val/test are split strictly by timestamp (`TimeSeriesSplitter` in
`forex_ml/data/splitting.py`) — never shuffled — so training data is always
chronologically before validation, which is always before test. `split_flow.py` also
purges `max(n_back, max_holding_bars)` bars on both sides of each split boundary: a
window reaches `n_back` bars backward and a triple-barrier label can reach
`max_holding_bars` bars forward, so without a purge gap the row right at a boundary
can have a window or label that overlaps the adjacent split. That's not leakage in
the sense of the model seeing future inputs at inference time, but the two adjacent
rows are highly autocorrelated, which can optimistically bias the validation/test
metric right at the seam (see Lopez de Prado's *purged k-fold CV* for the general
technique).

### The prediction target: triple-barrier labeling

The training target is **triple-barrier labeling**
(`forex_ml/data/triple_barrier.py`, wired into `TimeSeriesSplitter`), not a
fixed-horizon percent change. This replaced an earlier `pd_lead`/`volatility_lead`
scheme (both still computed in Stage 1 as diagnostic-only reference columns — see
below — but neither is trainable anymore).

Each row is labeled by running TWO independent first-passage "races" forward
bar-by-bar from the entry — one assuming a **long** position, one assuming a
**short** — each checked against its own *net-of-cost* return (one round-trip
spread charge, plus swap/rollover for every 5pm-New-York boundary actually
crossed, using that side's own swap rate). Whichever side's profit-take fires
first determines the label: long's profit-take firing (and short's not, or not
yet) → `+1`; short's profit-take firing (and long's not, or not yet) → `-1`; if
neither side's profit-take ever fires within `max_holding_bars` (either side hit
its own stop-loss, or both timed out) → `0`, flat. The label maps directly onto
the same class convention the rest of the pipeline (and `forex-strategy`'s
backtest) already assumes: `-1 → class 0` (short signal), `0 → class 1` (flat),
`+1 → class 2` (long signal) — no percentile-threshold fitting needed, since the
label is already discrete.

This is a genuine redesign, not the original scheme: earlier, only the long race
existed, and `-1` just meant "the long's stop-loss fired" — reused as a "short
signal" without ever independently confirming a short would actually have been
profitable. That was a real bug: long's stop-loss and short's true profitability
are NOT mirror images once cost enters — cost *adds to* a long's loss (so a
smaller drop stops it out), but cost *eats into* a short's gain (so it takes a
*bigger* drop for a short to genuinely clear its own profit-take). The old scheme
systematically overstated how easy it was for a short to win. Expect the fixed
scheme to produce a real class-balance shift toward more flat (`class 1`) labels
relative to the old one — a "short" label now requires independently confirmed
short profitability, not just "the long lost." That's the redesign doing its job
correctly, not a regression to chase. See `forex_ml/data/triple_barrier.py`'s
module docstring for the exact merge/tie-break rules (including how the rare
same-bar double-fire case is resolved by an explicit, tested rule rather than
assumed to be unreachable).

Four `params.yaml` knobs control it, under `split:`:

- `profit_take_pct` / `stop_loss_pct` — net-of-cost return thresholds (percent),
  shared by both the long and short race.
- `max_holding_bars` — the vertical barrier: give up on both races and label `0`
  after this many bars if neither side's profit-take was hit.
- `swap_cost_pct_per_night` — charged once per 5pm-New-York rollover boundary
  actually crossed (DST-aware). This is now a **fallback only**:
  `split_flow.py`/`rolling_cv.py` prefer real, live rates fetched from
  forex-etl's `swap-rate` InfluxDB measurement (see
  `forex_ml/data/swap_rates.py`), falling back to this constant (for both the
  long and short race) only if no live snapshot exists yet for the pair being
  trained. OANDA's `long_rate`/`short_rate` are annual rates as decimals (0.05 =
  5%/year, confirmed via OANDA's own v20 API docs), converted to a per-night
  percentage (`rate * 100 / 365`, a simple Actual/365 approximation) and
  sign-flipped so a real charge becomes a positive cost. Both sides are resolved
  and used now (`resolve_swap_cost_pct_per_night` returns a `(long, short)`
  tuple) — the short rate used to be discarded here, back when labeling was
  long-side-only.

**None of the four current threshold values (`profit_take_pct`/`stop_loss_pct`/
`max_holding_bars`/the `swap_cost_pct_per_night` fallback) have been empirically
validated** the way `n_back=200` was (via real ACF/PACF diagnostics) — they're
starting points to revisit once there's a real look at each pair's move-size
distribution, not a considered final answer. See `params.yaml`'s own comments for
the current values.

`pd_lead`/`spread_close_lead`/`volatility_lead` remain valid, useful **reference**
columns for diagnostics — `forex_ml.diagnostics.autocorrelation`'s `--column` flag
and `forex_ml.diagnostics.feature_impact`'s `--target` flag both default to
`pd_lead` (a plain hardcoded default now, not a "whatever the configured target is"
resolution, since there's no longer a single trainable column name to resolve to).

### Gradient and input clipping

`train.gradient_clip_norm` (default `1.0`, applied as `clipnorm` on the Adam
optimizer in `forex_ml/training/model.py`) bounds the gradient norm on every training
step. This was added after the first real training run at `n_back=200` against full
production history — loss diverged to NaN partway through the very first epoch, on
clean, NaN/Inf-free input data. The cause was unclipped gradients exploding through
deep backprop-through-time: `number_of_cells_per_rnn_layer` was `[300, 300, 300, 300,
300]` at the time (5 stacked LSTM layers — see "Layer depth" below for why it's 3
now), and `n_back=200` means each layer unrolls 200 timesteps — a lot of
multiplicative depth for gradients to blow up across if nothing bounds them.

This gap was **always there**, not something the modernization introduced — the
original `lstm.py`'s `compile_generic_regressor` had no gradient clipping either (see
`git show 4186e46:ML-engineering/forex-ML/lstm.py`). It simply never got exercised: the
original notebooks, and every test/config this pipeline had run before, used smaller
`n_back` values or smaller-scale data, shallow enough that exploding gradients never
actually happened to trigger. It took training on real, full-history data at the
pipeline's actual configured `n_back=200` to hit it for the first time.

Gradient clipping alone turned out not to be enough. A retry with `clipnorm=1.0` ran
12 clean epochs, then went to NaN again mid-epoch-13 — later, but not eliminated. The
reason: `clipnorm` only bounds the *backward* pass (already-computed gradients). If the
*forward* pass itself overflows to `Inf`/`NaN` — which is exactly what a single
extreme input value does after compounding through 200 unrolled LSTM timesteps across
5 stacked layers — the resulting gradient is `NaN` before clipping ever sees it, and
`clip(NaN)` is still `NaN`. This isn't hypothetical: the real, z-scored training data
has **thousands of values beyond 10 standard deviations** in every non-cyclical
feature (`volatility`, `return`, `diff_spread_close`, `diff_volume`), consistent with
the heavy tails (`return` kurtosis ≈ 17) found in the ACF diagnostics.

`train.input_clip_value` (default `10.0`) fixes this at the source: `ClipInputs`
(`forex_ml/training/model.py`) is a real `keras.layers.Layer` subclass — not a bare
`Lambda`, so it serializes/deserializes correctly through `keras.models.load_model()`,
since `ModelCheckpoint` saves and later reloads this exact architecture — inserted as
the *first* layer of the model, clipping every input to `[-10, 10]` before it ever
reaches the first LSTM. This is complementary to gradient clipping, not a replacement
for it: `clipnorm` still bounds how aggressively the model can react to a
clipped-but-still-large value; `input_clip_value` bounds how large that value can be
in the first place.

**Even both together don't fully eliminate the risk** — confirmed directly (2026-07-10)
via a controlled investigation after another real divergence: the same architecture
(5 stacked 300-cell LSTM layers, the depth in use at the time — see "Layer depth"
below), data, and seed diverged to NaN at wildly different, unpredictable points
(immediately, ~42% through an epoch, ~98% through, or not at all) across both looser
and tighter clip values, on both CPU and GPU. This looks like genuine, hard-to-fully-
eliminate numerical fragility inherent to many stacked LSTM layers unrolled 200 steps
deep on real fat-tailed financial data, not a bug traceable to either clip value
specifically — tightening `gradient_clip_norm`/`input_clip_value` was tried and
reverted; it didn't help. `train.py` now treats this as an accepted risk to *recover*
from rather than fully prevent:
- `TerminateOnNaN` stops training immediately on the first NaN/Inf loss, instead of
  grinding through up to `early_stopping_patience` more full epochs of wasted compute
  on an already-unrecoverable model.
- A hand-rolled sub-epoch `_BatchCheckpoint` callback saves weights every 200 batches
  (whenever training loss is finite and improves on the best seen so far), so a run
  that diverges before completing its first epoch — which is when `ModelCheckpoint`'s
  ordinary epoch-boundary "best weights" safety net has nothing to fall back to — can
  still recover most of that epoch's real progress instead of producing a fully
  unusable model. `train_and_evaluate` checks the recorded loss history (not just the
  model's weights — a diverged run can produce `loss: inf` with every individual
  weight still finite) and reloads this checkpoint if needed, tagging the MLflow run
  with `recovered_from_batch_checkpoint` either way so this is visible rather than
  silently indistinguishable from a normal run.

A clean run is achievable — verified directly — just not guaranteed on the first
attempt.

### Layer depth

`train.number_of_cells_per_rnn_layer` is `[300, 300, 300]` (3 stacked LSTM layers) —
reduced from `[300, 300, 300, 300, 300]` (5 layers) on 2026-07-10, after a rolling-CV
depth comparison (`forex_ml.evaluation.rolling_cv`, 3 folds each, width held fixed at
300, EUR/USD H1) turned up a real sweet spot rather than a monotonic "fewer is
better" trend:

| Depth | LSTM test accuracy (mean ± std across folds) | vs. majority baseline (0.368) |
|---|---|---|
| 5 layers | 0.367 ± 0.045 | essentially tied — arguably losing |
| **3 layers** | **0.379 ± 0.020** | clearly ahead, and the most consistent fold-to-fold |
| 2 layers | 0.346 ± 0.075 | worse than baseline, and the least consistent (0.264–0.413 spread) |

5 layers was too deep — barely clearing (arguably losing to) the trivial majority-
class baseline, with real fold-to-fold noise. 2 layers was too shallow — worse than
baseline on average *and* by far the least stable of the three. A naive "just cut
depth" guess could easily have landed on 2 layers, which turned out to be the worst
option here, not the best — exactly the kind of thing empirical testing over guessing
is meant to catch. 3 layers also trains roughly 2.6x faster than 5, and — though not
separately isolated in this comparison — fewer stacked layers plausibly reduces the
numerical fragility described above too, since it directly shortens the
compounding-through-depth chain implicated there.

This used a scaled-down diagnostic (3 folds, ~8,000-bar training windows, not the
full ~11-year history) to keep runtime bounded on a single workstation — a real
directional signal, not the same statistical weight as a full-scale study. Like
`profit_take_pct`/`stop_loss_pct`/`max_holding_bars`, this hasn't been re-validated
against the full production dataset.

### Training instability: seed-dependent mode collapse (2026-07-13)

A real full EUR/USD H1 training run scored 0.428 accuracy, clearing both
baselines with a strong significance test. Re-running the *exact same*
architecture/data with only `tensorflow_seed` changed collapsed to predicting
"long" on 99% of rows and scored *below* the majority baseline. An audit of all
33 training runs logged this session (free — no retraining needed, just reading
already-logged MLflow artifacts) confirmed this is real and pervasive: ~39% of
all runs collapsed to one dominant class, and only 2 of those 13 collapses
traced to an already-understood NaN-divergence-and-recovery artifact — 11 more
happened during completely normal, non-diverged training.

Four architecture/regularization changes were tested via matched 5-seed
rolling-CV comparisons (same 8,000-bar fold, same seeds, one variable changed
at a time) to try to fix it:

| Change | Mean accuracy | Mode-collapse rate | NaN-divergence rate | Verdict |
|---|---|---|---|---|
| (baseline: current architecture) | 0.378 | 2/5 (40%) | 1/5 (20%) | — |
| Last LSTM layer `return_sequences=False` (see below) | 0.340 | 5/5 (100%) | 4/5 (80%) | **reverted** — made everything worse |
| Remove softmax-output `activity_regularizer` bug (see below) | 0.355 | 2/5 (40%) | — | **kept** — correct fix, but no measurable effect |
| `class_weight` (inverse-frequency balanced) | 0.331 | 1/5 (20%) | 2/5 (40%) | **kept as shipped default** — real trade-off, not a clean win |
| Revert 10x drift in `l1_regularization_constant` (1e-4 → original 1e-5) | 0.318 | 5/5 (100%) | 5/5 (100%) | **rejected** — worst result of the four |

Two real bugs were found and fixed along the way, independent of whether they
turned out to explain the instability:

- `build_lstm_regressor` had every LSTM layer, including the last, configured
  with `return_sequences=True` — inherited verbatim from the original
  `lstm.py` (confirmed via `git show 4186e46:...lstm.py`). Since the final
  layer never collapsed to a single summary vector, `Flatten()` was flattening
  the entire `(n_back=200, units=300)` sequence into a 60,000-scalar vector
  before a 7-unit Dense layer, instead of the standard "last recurrent layer
  returns only its final timestep" pattern. A well-reasoned hypothesis for the
  instability — tested and empirically refuted (table above). The likely
  explanation: that huge, redundant flatten may have been acting as an
  accidental stabilizer (a large, forgiving optimization landscape with many
  mediocre-but-not-degenerate solutions), and removing it concentrated the
  network into something more prone to sharp, degenerate convergence.
- The final output Dense layer (softmax) had `activity_regularizer=L2(...)` —
  in Keras this penalizes a layer's own returned output, meaning it was
  directly regularizing the model's output *probabilities* toward zero, not
  its weights. This directly explains an earlier finding that the model never
  assigned more than 50% confidence to anything in a real backtest. Fixed
  (removed from the final layer only); had no measurable effect on the
  collapse/confidence numbers above, but is simply the correct thing to do
  regardless.

None of the four changes resolved the underlying instability. A follow-up
comparison against a gradient-boosted-tree model (`sklearn.ensemble.
HistGradientBoostingClassifier`, since `xgboost`/`lightgbm` aren't installed)
on the exact same engineered features (just the current bar, not the full
200-step sequence) and the same bidirectional triple-barrier labels found the
instability is specific to this LSTM architecture, not an absence of signal in
the data: GBT was stable across every seed tried (accuracy 0.383–0.384, zero
mode collapse), consistently beating both baselines. A promising cost-aware
backtest result at one confidence threshold (win rate 0.528, net +2.27%, the
first positive net P&L of the whole investigation) did not survive a proper
significance check (p=0.092) or a 5-fold multi-window validation (pooled win
rate never significantly exceeded 50% across any threshold) — so GBT's
demonstrated advantage so far is stability, not a validated trading edge. Full
narrative: `blog-posts/13-our-heroine-trades-her-thoroughbred-for-a-mule.md`.

### Chasing a trading edge: features, pairs, and a regression target (2026-07-14)

Three follow-up questions from the mode-collapse investigation above, taken in
turn:

**Does a shorter memory fix the LSTM's instability?** `n_back` was cut from 200
to 50 (same architecture, same everything else). NaN divergence disappeared
entirely (0% vs. 20% of seeds), but mode collapse got *worse* (100% vs. 40% of
seeds) and mean accuracy dropped well below the (also-shifted, smaller-window)
majority baseline. Rejected — trades one failure mode for a worse one, not a fix.

**Does GBT's stability and full-window-feature advantage generalize beyond
EUR/USD?** Surveyed GBP/USD, USD/JPY, and AUD/USD at H1, 3 seeds each, both
feature variants (current-bar-only vs. the full flattened `n_back`-length
window). Findings held up across every pair: GBT was stable everywhere (no
mode collapse, low seed-to-seed variance), the full-window feature variant
beat current-bar-only on every single pair tested, and a spot-check of the
LSTM on these same new pairs reproduced its instability there too (divergence
on the new pairs' runs). Conclusion: both the LSTM's fragility and GBT's
stability are properties of the architecture/model class, not something
peculiar to EUR/USD — and GBT should always use full-window features going
forward, not current-bar-only.

**Would predicting the actual expected return, instead of a 3-class
short/flat/long label, produce a better-aligned trading signal?** Motivation:
classification optimizes for getting the direction right, not for the
magnitude of money made or lost, which is a mismatch with the actual goal.
Regressing `net_return_pct` (the cost-adjusted, triple-barrier-bounded realized
return — bounded, unlike raw return, which risks the model collapsing to
predicting the unconditional mean under low signal-to-noise, a regression
flavor of the classifier's mode-collapse problem) with GBT (full-window
features) and trading only when the predicted return exceeded a threshold
produced an eye-catching single-window result: at threshold=0.30, win rate
0.526, net **+20.32%**, p=0.018 (n=1,621 trades) — the best single number the
whole investigation had produced.

It did not survive being asked twice. The same 5-fold multi-window validation
used to debunk the classifier's confidence-filtering result was run against
this regression approach, same fold sizing (10,000/2,000/2,000 bars, sliding):

| threshold | total trades | pooled win rate | p-value | total net P&L | per-fold win rates |
|---|---|---|---|---|---|
| 0.00 | 6,495 | 0.488 | 0.974 | -1.82% | 0.556, 0.413, 0.502, 0.475, 0.495 |
| 0.05 | 6,487 | 0.488 | 0.978 | -3.35% | 0.556, 0.412, 0.502, 0.474, 0.495 |
| 0.10 | 6,448 | 0.488 | 0.973 | -3.20% | 0.555, 0.413, 0.502, 0.474, 0.496 |
| 0.15 | 6,252 | 0.487 | 0.982 | -12.96% | 0.551, 0.413, 0.505, 0.470, 0.498 |
| 0.20 | 5,567 | 0.485 | 0.989 | -37.55% | 0.559, 0.401, 0.503, 0.468, 0.492 |
| 0.30 | 1,381 | 0.498 | 0.564 | -30.48% | 0.586, 0.331, 0.482, 0.458, 0.465 |

The specific window that produced the promising result is fold 0 in this
table (win rate 0.556–0.586 across thresholds) — genuinely good, but offset by
a genuinely bad fold (fold 1: 0.331–0.413) and three folds hovering near a
coin flip. Pooled, no threshold clears 50% with any significance, and net P&L
is negative at every threshold except the trivial (still slightly negative)
threshold=0. Same verdict as the classifier's confidence-filtering result:
one favorable window, not a validated edge.

**Where this leaves the model-selection question**: GBT remains the more
trustworthy model class (stable, generalizes across pairs, always beats
baselines on the classification task) — but neither approach tried so far for
turning that into a trading signal (confidence-filtered classification,
return-threshold regression) has produced an edge that survives multi-window
validation. The open problem is no longer "is the model unstable" — it's
solved. It's "does this feature set, at this granularity, contain a
profitable edge at all," which remains unanswered.

### Richer features: multi-timeframe, volatility regime, momentum, cross-pair (2026-07-14)

With "does this feature set contain a profitable edge" still unanswered, the next
lever tried was the feature set itself: every model so far had only ever seen a
trailing window of single-bar-derived features (volatility, return, spread/volume
diffs, three short moving-average lookbacks, calendar/session flags) — nothing
capturing longer-horizon trend, volatility-regime shifts, classic oscillators, or
other pairs' behavior. Three rounds of new features were added to `forex_ml.data.
features`, each tested (135+ unit tests passing throughout) and screened via the
same single-window GBT check used for every other idea this session, before
deciding whether a 5-fold multi-window validation was even warranted:

| | Original (10 feat) | +multi-timeframe/vol-regime (13) | +momentum (16) | +cross-pair (17) |
|---|---|---|---|---|
| Classification test acc | ~0.41 | 0.411 | 0.413 | 0.412 |
| Classification backtest, best result | win 0.528, net **+2.27%** | win 0.482, net -79.9% | win 0.492, net -65.6% | win 0.509, net -11.8% |
| Regression backtest, threshold=0.30 | win 0.526, net **+20.32%** | win 0.505, net +8.91% | win 0.507, net -2.10% | win 0.477, net -42.13% |

- **Multi-timeframe trend + volatility regime**: extended `ma_lookback_list` with a
  96-bar (~4 trading day) lookback, giving `return_MA_96` (a longer-horizon trend
  signal, for free via the existing moving-average machinery) and `volatility_MA_96`
  (a long-horizon volatility baseline). A new `add_volatility_regime_features`
  computes `volatility_regime_ratio = volatility_MA_12 / volatility_MA_96` — derives
  which two lookbacks to use from `min`/`max(ma_lookback_list)` rather than
  hardcoding, so it doesn't break under a scaled-down test config, and falls back to
  a neutral 1.0 rather than dividing by a near-zero long-window volatility.
- **Momentum/oscillator analogs**: `add_momentum_features` adds three
  stationarity-safe analogs of classic price-momentum indicators, built on the
  already-stationary `return` column rather than raw price levels (this project's
  existing no-raw-price-features convention) — `return_sma_crossover` (an
  SMA-crossover MACD analog), `return_zscore_12` (a Bollinger-band-position analog),
  and `rsi_12` (a simple, non-Wilder-smoothed RSI variant, since Spark window
  functions can't express Wilder's recursive smoothing without a per-partition UDF).
- **Cross-pair "USD strength"**: `compute_cross_pair_usd_strength` averages
  sign-adjusted returns across this project's other configured pairs (every pair is
  USD-quoted or USD-based, so this approximates broad-dollar strength/weakness
  independent of the target pair's own price action) into one
  `usd_strength_return` feature. Implemented as a new `pull_cross_pair_return_task`
  in `prepare_data_flow.py` that pulls each other pair's raw candles fresh (same
  date range) and reduces to just `return`, rather than depending on that pair's own
  already-materialized Stage-1 output — avoids an ordering dependency where pair A's
  prep would require pair B's Stage-1 to already exist. `add_cross_pair_features`
  left-joins this onto the target pair by timestamp, falling back to a neutral 0.0
  for any timestamp with no cross-pair data (a feed gap) rather than dropping the
  row.

Cross-pair was the one addition that moved something real: classification win rate
crossed 50% at three separate confidence thresholds for the first time since the
original feature set, and the best result got much closer to breakeven (-11.8% net
vs. -65.6% without it). But it never turned net-positive — transaction costs still
dominate every threshold — so per the same discipline that caught the earlier
false-positive results (only validate a result across multiple windows if it's
already promising, i.e. net-positive, in one), this didn't clear the bar for the
expensive 5-fold check. Regression got worse with cross-pair added, not better.
`FeatureParams.engineered_columns` (the hand-maintained mirror Stage 1's columns
are checked against — see Configuration above) gained a two-directional test
(`test_engineered_columns_matches_actual_stage1_output`) while this code was being
touched, closing a real gap where only one direction of that mirror was ever
checked.

**Net verdict**: 12 new, tested, real features across three phases, and none
produced a profitable single-window result on EUR/USD H1. Cross-pair is the most
encouraging direction found so far (worth revisiting, e.g. with a longer cross-pair
lookback/smoothing rather than a same-bar snapshot) — but "richer features" as a
category hasn't cracked profitability at this pair/granularity yet.

### Backtest fix: pricing a trade by the side actually taken (2026-07-14)

A double-check of `forex_strategy.backtest.simulate_trades` (prompted by an
attempt to explain why every backtest's `cost_pct` so heavily outweighed
`gross_pnl_pct`) turned up a real, previously-unflagged imprecision: `triple_
barrier.py` runs two independent races per row (long and short) but used to
persist only ONE side's outcome — whichever won, or resolved first if flat
(`exit_bar_offset`/`raw_return_pct`). When a model's prediction *disagreed* with
the label — roughly half of every trade in a near-50%-win-rate backtest — the
backtest had no choice but to price that trade using the *other* side's real
outcome as a stand-in, not the true result of the position actually taken.

Quantified directly against real EUR/USD H1 data (14,393 rows, both races
recomputed) before fixing anything: the worst case — both sides genuinely
winning, so the discarded side's real profit gets thrown away entirely — never
occurred (0/14,393). The typical mispricing on a wrong-direction trade was small
(mean |gap| ≈ 0.027–0.028 percentage points, median exactly 0, since the two
races usually resolve at the same bar under symmetric profit-take/stop-loss
thresholds), with a real but modest tail. Not large enough to explain the
session's wide win-rate swings, but a genuine, fixable gap.

**Fix**: `TripleBarrierLabels` (and `Splits.test`/the predictions.npz artifact)
now persist each side's own true outcome independently —
`long_raw_return_pct`/`long_exit_bar_offset` and
`short_raw_return_pct`/`short_exit_bar_offset` — alongside the existing merged
single-winner view (`raw_return_pct`/`exit_bar_offset`, still used for the
regression target and persistence baseline, which legitimately want "the
resolved outcome," not a per-direction split). `simulate_trades` now takes
`long_raw_return_pct`/`short_raw_return_pct` (and
`long_exit_timestamp`/`short_exit_timestamp` for swap/rollover accounting)
instead of a single array, and selects the side matching each row's own
predicted position before computing P&L — so a wrong-direction trade is priced
by what would truly have happened, not a substitute.

Re-running the corrected backtest against the same model/features used for the
cross-pair result above showed the expected small, real shift — not a dramatic
reversal, confirming the fix matters without overturning this session's
conclusions:

| min_confidence | win_rate (before fix) | win_rate (after fix) | net_pnl_pct (before) | net_pnl_pct (after) |
|---|---|---|---|---|
| 0.00 | 0.495 | 0.482 | -69.54 | -69.15 |
| 0.40 | 0.502 | 0.489 | -43.18 | -45.50 |
| 0.45 | 0.503 | 0.489 | -37.54 | -42.25 |
| 0.50 | 0.493 | 0.475 | -40.69 | -46.22 |
| 0.55 | 0.509 | 0.489 | -11.76 | -16.75 |

### Granularity comparison: EUR/USD at H4 (2026-07-14)

With the backtest confirmed trustworthy, one more lever was tried before
concluding the search: a coarser input granularity, on the theory that H1 might
simply be too noisy a timescale relative to real trading costs. H4 wasn't
available in this pipeline at all before this — added as a new collection
granularity in the sibling `forex-etl` project (see that repo's README), which
also surfaced and fixed an unrelated, real bug: `ForwardFillInator`'s
gap-filling grid assumed a fixed number of UTC seconds between bars forever,
which breaks for any granularity anchored to local time-of-day (H4, D) once
DST shifts the real boundary by an hour — confirmed against real history (H4:
~66% of rows misaligned before the fix, 0% after, across all 17 years on
file) and fixed there.

Daily granularity was considered first and rejected: only ~3,000 bars total
for EUR/USD since 2015, far too few to run the same 5-fold, 10,000+-bar
validation discipline used throughout this investigation without badly
under-powering it — exactly the kind of tradeoff (bigger typical moves per
bar vs. far fewer independent bars to prove it with) flagged when the
granularity question first came up.

H4 config was scaled directly off the production H1 config (every lookback,
holding period, and barrier divided/multiplied by 4, so the comparison
isolates "granularity of the input bars," not a different holding-period
philosophy) rather than independently re-validated via ACF/PACF the way
n_back=200 was for H1. Same GBT methodology as EUR/USD H1 (full-window
features, classification and net_return_pct regression), including the same
three-round feature-richness progression (multi-timeframe/volatility-regime,
momentum, cross-pair):

| | Classification net P&L (best threshold) | Regression net P&L (best threshold) |
|---|---|---|
| Baseline (10 feat) | -0.42% (conf 0.50, win 0.497, n=197) | -6.63% (thr 0.30, win 0.486) |
| +multi-timeframe/vol-regime | -4.92% (conf 0.45, win 0.500, n=532) | -48.64% (thr 0.00, win 0.465) |
| +momentum | -6.54% (conf 0.40, win 0.506, n=814) | -12.96% (thr 0.30, win 0.458) |
| +cross-pair | +0.05% (conf 0.45, win 0.507, n=534) | -10.90% (thr 0.30, win 0.469) |

Regression got uniformly worse with richer features at every round, mirroring
H1. Classification is noisy round to round with no clean trend; the one
near-zero positive number (+0.05%, essentially exactly breakeven) is far
weaker evidence than the H1 cross-pair result that already failed 5-fold
validation (that one had win rate clearing 50% by several points at three
separate thresholds, not a fraction of a percentage point at one) — not worth
the compute of a multi-window check on a signal this marginal, on a test set
(2,651 rows) already a quarter the size of H1's. Verdict: the coarser-timescale
hypothesis doesn't fare any better than H1 did.

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

Every command here (and every diagnostic in the next section) that touches Spark
exposes `--spark-memory` (default `70g`, applied identically to
`spark.driver.memory`/`spark.executor.memory`/`spark.driver.maxResultSize` — see
`forex_ml/spark_session.py`), rather than hardcoding a value — the right amount
depends on the machine actually running it, and `70g` only ever made sense on the
one workstation the original notebooks were written for.

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
`baseline_persistence_test_accuracy` (predict this period repeats the most recent
*already-resolved* prior period's actual class) — so `test_accuracy` is never read
in isolation.

**`baseline_persistence_test_accuracy` had a real information-leakage bug, fixed
2026-07-10.** It used to predict row `i` using row `i-1`'s actual label
unconditionally — but under triple-barrier labeling a label isn't resolved until
its barrier race actually finishes (a mean of ~15 bars out of `max_holding_bars=24`
on H1 data), so that "previous row" was still unresolved for ~98.5% of rows,
measured directly on a real EUR/USD H1 test split — the baseline was borrowing up
to 23 bars of future information. It scored ~0.856 that way; the fixed version
(persist only from a prior row once its label has genuinely resolved, tracked via
`exit_bar_offset`) scores ~0.388 — barely above the majority baseline, not the
dominant baseline the old number implied. Historical MLflow runs logged before this
fix keep their old (inflated) number; only new runs use the corrected calculation,
and `forex_ml.evaluation.multiple_comparisons.report_across_pairs` skips any
pre-fix run when comparing against the persistence baseline rather than silently
misaligning old- and new-format `predictions.npz` artifacts. Every run
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

### Finding "the model for (instrument, granularity)"

Every pair registers under the same shared `train.mlflow_experiment_name` — the
Model Registry has no per-pair identity of its own. Every registered model version is
tagged at registration time with `instrument`, `granularity`, `config_signature`
(the same hash `forex_ml/evaluation/multiple_comparisons.py` uses to group runs by
configuration), and `column_y` (which target this version was trained on — always
`"triple_barrier"` for a current training run), so the right version can be found
via `MlflowClient.search_model_versions` without grepping the source run's logged
params:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="sqlite:///mlflow.db")
versions = client.search_model_versions(
    "name = 'forex-lstm' and tags.instrument = 'EUR/USD' and tags.granularity = 'H1'"
    " and tags.column_y = 'triple_barrier'"
)
```

`column_y` was originally added to tell `pd_lead`- and `volatility_lead`-trained
models apart when both could share a pair — moot now that triple-barrier labeling
is the only trainable target, but the tag stays: it's what lets a consumer reliably
exclude older, pre-migration model versions (registered before this change, still
tagged `pd_lead`/`volatility_lead` or untagged entirely) from a search that expects
the current scheme, without grepping every candidate version's source run.

### Backtesting support

The `<run_uid>_predictions.npz` artifact (see above) also carries the raw softmax
probabilities (`lstm_pred_proba`) and the test split's timestamp/price/spread/raw
target value/exit timing/realized volatility (`test_timestamp`/`test_price`/
`test_spread`/`test_y_raw`/`test_exit_bar_offset`/`test_realized_volatility`)
alongside the existing correctness booleans used for McNemar's test. A
correct/incorrect boolean is enough to compare two classifiers, but a real backtest
(the sibling [`forex-strategy`](../forex-strategy) project) needs to know how
confident the model was, at what price and spread cost, what actually happened
(the realized % move, not just which barrier it hit), how long the trade actually
took to resolve, and how volatile the market recently was (for position sizing),
to compute actual P&L, not just accuracy.

Also carried: `test_long_raw_return_pct`/`test_long_exit_bar_offset` and
`test_short_raw_return_pct`/`test_short_exit_bar_offset` — each side's OWN true
triple-barrier race outcome (see "Backtest fix" above and
`forex_ml.data.triple_barrier.TripleBarrierLabels`), independent of which side the
label actually was. `test_y_raw`/`test_exit_bar_offset` only reflect whichever side
won; a backtest needs the long/short-specific fields to correctly price a
prediction that disagrees with the label.

Every run also logs `column_y` as a param — always `"triple_barrier"` for real
training runs now, kept as an explicit logged value (not hardcoded away) so a
downstream consumer can still confirm what `y_raw`/`test_y_raw` is before treating
it as a directional quantity, and so `multiple_comparisons`'s config-signature
grouping keeps working exactly as it did when `column_y` distinguished `pd_lead`
from `volatility_lead` runs.

The same fields are available directly on `Splits.test` (see
`forex_ml/data/splitting.py`) — `price`/`spread`/`realized_volatility` via
`COLUMNS_PASSTHROUGH` in `forex_ml/data/features.py` (`mid_close`/`spread_close`/
`realized_volatility` are the reference columns kept around after feature
engineering, explicitly excluded from `columns_x` so they're never fed to the model
as input); `y_raw` is `raw_return_pct` (the *pre-cost* realized return at the row's
actual exit bar — deliberately not `net_return_pct`, which is already net of
spread/swap and would double-count cost if fed to a backtest that charges its own);
`exit_bar_offset` is how many bars the label actually took to resolve, letting a
backtest compute a real, variable holding period instead of assuming a fixed one;
and `realized_volatility` is a fixed 12-bar backward-looking rolling average of
single-bar `volatility` (`mid_high - mid_low`), computed in `add_market_features` —
real, already-observed recent volatility for a backtest to size positions against,
replacing the old approach of training a second model to predict a
`volatility_lead` ordinal class. Only the **test** split carries any of this
(`train`/`val` stay exactly `{"M", "y"}`) since backtesting only ever needs to
reconstruct P&L on the held-out set.

`long_raw_return_pct`/`long_exit_bar_offset` and
`short_raw_return_pct`/`short_exit_bar_offset` are also on `Splits.test`, straight
from `TripleBarrierLabels` — each side's own true race outcome, needed so
`forex_strategy.backtest.simulate_trades` can price a trade in either direction
correctly rather than substituting the winning side's outcome for a
wrong-direction prediction (see "Backtest fix" above).

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
uv run python -m forex_ml.diagnostics.feature_impact --instrument EUR/USD --granularity H1
```

Quick, linear-approximation screening for "which time series most impacts the
target" (`forex_ml/diagnostics/feature_impact.py`) — a repeatable tool for
evaluating a candidate column **before** spending a training run on it, or for
deciding what to drop from `split.columns_x` if GPU memory becomes the binding
constraint. `--candidates` accepts ANY comma-separated list of Stage-1 columns, not
just the ones currently in `columns_x` — the whole point is to be able to check a
column that isn't in production config yet. Four techniques, cheapest and least
rigorous first:

- **Cross-correlation (CCF)** — correlation between each candidate `lag` bars in the
  past and the target now, across a range of lags. Direct extension of the ACF/PACF
  machinery above to a candidate-vs-target pair. Cheapest, fewest assumptions.
- **Pairwise Granger causality** — does a candidate's own history improve a linear
  forecast of the target beyond the target's own history, at one shared lag across
  every candidate (not a scanned range per candidate — that would be its own
  uncorrected multiple-comparisons problem stacked on the one already being
  corrected for across candidates). BH-FDR corrected across candidates, same reuse
  of `forex_ml.evaluation.multiple_comparisons` as the pair-comparison report.
  Flags each candidate's stationarity verdict too, since Granger validity assumes it.
- **VAR + block-exogeneity Wald tests + forecast-error variance decomposition
  (FEVD)** — the properly multivariate version of Granger causality, controlling for
  every other candidate jointly instead of testing pairs in isolation. Pairwise
  Granger can't see multicollinearity; VAR can. FEVD reports what fraction of the
  target's forecast-error variance each candidate accounts for at a given
  horizon — the closest linear analogue of a feature-importance ranking for a
  time-series system.
- **Lasso-regularized lagged regression** — a single-equation distributed-lag model
  (target ~ every candidate at lags 0..max_lag) with L1 regularization, which shrinks
  unhelpful lags toward (not always exactly to) zero — a direct, automatic
  "drop this" signal. Uses `TimeSeriesSplit`, not the sklearn default k-fold, to pick
  the regularization strength — ordinary k-fold would leak future folds into past
  ones, the same chronological discipline as everywhere else in this pipeline.

All four are explicitly linear approximations — an LSTM can exploit nonlinear and
cross-feature structure none of them can see. This is a floor on which candidates are
worth a full training run, not a ceiling on what could possibly matter, the same
relationship the ACF/PACF diagnostic has to `n_back`.

**A real gotcha this tool checks for, not just a theoretical risk**: a fixed-period
sin/cos encoding (`day_sin`/`day_cos`, `week_sin`/`week_cos`) is *exactly* linearly
dependent across lags — `sin(ω(t-k))` is an exact linear combination of `sin(ωt)` and
`cos(ωt)` for any fixed lag `k` (the angle-subtraction identity), so including many
lags of such a pair in a VAR adds zero information and can wreck its numerical
conditioning. Confirmed directly on real data: an 11-lag block of `day_sin`/`day_cos`
alone has rank 2 of 22 columns, condition number ~1.6×10¹². When this happens, the
VAR causality report's `rank_warning` field (and the printed `WARNING:` line) says so
explicitly — a candidate showing a large FEVD share but "not significant" in the
causality test is a sign to check this warning, not a sign the candidate doesn't
matter.

**Cross-pair candidates**: every technique above also works with candidates drawn
from a DIFFERENT instrument than the target — e.g. does GBP/USD's `return` help
predict EUR/USD's `volatility_lead`? No new ingestion needed, since all 7 major
pairs already flow through the same Stage 1 pipeline, each under its own
`(instrument, granularity)` key:

```bash
uv run python -m forex_ml.diagnostics.feature_impact --instrument EUR/USD --granularity H1 \
    --cross-pair-candidates "GBP/USD:return,diff_spread_close;USD/JPY:volatility"
```

`--cross-pair-candidates` is one semicolon-separated group per candidate
instrument, columns within a group comma-separated (parsed by
`_parse_cross_pair_candidates`); it takes over from `--candidates` when given.
`load_cross_pair_target_and_candidates` renames each candidate column to
`{instrument}__{column}` before joining on `unix_epoch_s` (an inner join — pairs
can have slightly different available timestamps, e.g. differing forward-fill
history), since every pair's Stage 1 output uses the same column names and would
otherwise collide. Once loaded and renamed, every report function is
candidate-source-agnostic — `analyze_cross_pair_feature_impact` is otherwise
identical to the single-pair `analyze_feature_impact`.

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
defaults to `max(feature.n_back, split.max_holding_bars)` from `params.yaml`
(overridable), purging that many bars on both sides of each fold's train/val and
val/test boundary so
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

## Cost-aware labeling (triple barrier) — module reference

This is the module (`forex_ml/data/triple_barrier.py`) behind "The prediction
target: triple-barrier labeling" under Configuration above — see that section for
how it's wired into `TimeSeriesSplitter`/production training. This section is a
quick reference for using it directly (e.g. ad-hoc analysis outside the normal
Stage 2 flow):

```python
from forex_ml.data.triple_barrier import triple_barrier_labels_from_frame

labeled = triple_barrier_labels_from_frame(
    df,  # Stage-1 df_non_time_series, sorted by unix_epoch_s for one pair
    profit_take_pct=0.5, stop_loss_pct=0.3, max_holding_bars=8,
    long_swap_cost_pct_per_night=0.02,   # e.g. the negative of a SwapRateRecord long_rate, if negative
    short_swap_cost_pct_per_night=0.01,  # e.g. the negative of a SwapRateRecord short_rate, if negative
)
```

For real values instead of guessed ones, `forex_ml.data.swap_rates.fetch_current_swap_rates(instrument)`
returns `(long, short)` already converted from OANDA's raw annual-rate-as-decimal
convention to this per-night percentage — pass both straight through.

Swap/rollover is charged once per 5pm New York rollover boundary *actually crossed*
between entry and exit — computed DST-aware in local `America/New_York` time
(`count_rollovers_crossed`), not the fixed-UTC approximation the trading-session
features above use, since an hour's error here is the difference between being
charged a night's swap or not, not just a soft diurnal-pattern approximation. An
intraday (H1/M15) hold usually crosses zero rollovers; multi-day holds accumulate
one charge per night actually held through, not one per bar.

Bidirectional: two independent races (long, short) run per row, each against its
own swap-cost input — see "The prediction target: triple-barrier labeling" above
for the merge/tie-break rules, and `triple_barrier.py`'s module docstring for the
full detail (including why a same-bar double-fire needs an explicit tie-break
rather than being assumed impossible, and why `spread` must be non-negative).
Both races' own outcomes are returned, not just the merged winner: `labeled`
carries `long_exit_bar_offset`/`long_raw_return_pct` and
`short_exit_bar_offset`/`short_raw_return_pct` alongside the single-winner
`exit_bar_offset`/`raw_return_pct` — see "Backtest fix" above for why a backtest
needs both.

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
correction, rolling CV, and triple-barrier labeling — each with at least one real
end-to-end test (real Spark-engineered Stage-1 output and/or a real local MLflow
store), not just unit tests against hand-built arrays.

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
