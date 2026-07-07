# Data Scientist vs. The Villainous Jupyter Notebooks of Pipeline Spagetti

Our heroine — inauspicious data scientist by day, remix crusader by night, and, this week, a woman staring into the abyss of a currency-trading LSTM built entirely out of three Jupyter notebooks and a prayer — had a decision to make.

The notebooks worked, in the sense that "worked" means "ran, in the correct order, on
the one machine that still remembered which environment variables it needed." There
was a README. The README said, essentially, *run these three notebooks, in this
order, and do not think too hard about it.* There was a loop that used `papermill` to
execute one of them over and over for different currency pairs, dutifully overwriting
its own output every single time, because the output path never changed no matter
which pair it was working on. There was a single 684-megabyte pickle file standing in
for what should have been fourteen. Our heroine looked at all of this the way one
looks at a load-bearing shelf held up by a single, confident-looking nail, and thought:
*we are going to need a bigger nail.*

She did not, in fact, get a bigger nail. She got MLflow, Prefect, DVC, pydantic, and
approximately one hundred percent more self-respect.

**The Notebooks, Reconsidered**

The original pipeline pulled candlestick data from InfluxDB, engineered features with
PySpark, windowed it into sequences, and trained an LSTM to guess whether the next
candle would go up, down, or sideways. None of that was wrong, exactly. It's just that
"none of that was wrong, exactly" is a sentence you can only say about code you have
never tried to run twice, on two different machines, six months apart, expecting the
same answer both times.

So the notebooks became a real installable Python package, `forex_ml`, with the
actual logic — feature engineering, splitting, training — living in ordinary,
testable, importable modules. The notebooks themselves got demoted to
`notebooks_legacy/`, kept around for ad-hoc poking-around purposes only, like a
retired mascot costume nobody has the heart to throw away.

**Config as a First-Class Citizen, Not a Dict Someone Typed at 2 AM**

Every parameter that used to live as a bare dictionary literal scattered across three
notebooks — instrument lists, feature windows, split proportions, model
hyperparameters — now lives in exactly one place, `params.yaml`, validated at load
time by a set of `pydantic` models. This sounds like a small thing until you consider
what it replaces: a config that references a feature column Stage 1 never actually
produces used to fail three stages downstream, with a shape-mismatch error that told
you nothing except that something, somewhere, was sad. Now it fails immediately, at
load time, with a message that names the exact offending column. Our heroine
considers this one of the more satisfying trades she has ever made — a few extra
lines of validation code for the permanent right to never again play "guess which of
my six typos broke the pipeline."

**MLflow: So Nothing Has to Be Taken on Faith**

Every training run now logs its hyperparameters, its per-epoch metrics, and a real
held-out test evaluation to MLflow, and registers the resulting model in MLflow's
Model Registry. This replaced a genuinely alarming bug in the original `lstm.py`: the
notebook trained using Keras's `validation_split=0.2`, which quietly re-slices a
*fresh* validation set out of whatever you hand it — meaning the actual, carefully
time-ordered validation split the earlier notebook had computed was pickled to disk,
loaded back up, and then completely ignored. There was also no held-out test
evaluation of any kind. The model trained, declared victory, and nobody ever checked
its work against data it hadn't touched during training. This is now fixed, logged,
and — this part matters — impossible to quietly regress back into, because a test
would notice.

**Prefect, Because the ETL Side Already Had Opinions**

The sibling `forex-etl` repository already used Prefect for its own pipelines —
`@flow`/`@task` decorators, retries around I/O, structured logging via
`get_run_logger()`. Rather than introduce a second orchestration tool with its own
conventions, the new `forex_ml/flows/` package mirrors the existing ones exactly:
`prepare_data_flow`, `split_flow`, `train_flow`, and an optional scheduled
`serve.py` for unattended weekly retraining. One orchestrator, one set of
conventions, one fewer thing for future-heroine to relearn from scratch.

**DVC, for When "It Worked On My Machine" Isn't Good Enough**

`dvc.yaml` now defines a `prepare → split → train` chain per `(instrument,
granularity)` pair, with `params.yaml` as the single source of truth DVC also reads
for cache invalidation — change a hyperparameter, and only the stages that actually
depend on it re-run. Every output path is keyed on `(instrument, granularity, n_back,
lookahead)`, which sounds pedantic until you remember the alternative was one shared
pickle file that could only usefully represent one currency pair's data at a time.
Fourteen pairs, one pickle, musical chairs nobody wanted to play — solved, in the
most boring way possible, by just naming the files correctly.

**What This Modernization Actually Bought**

Not elegance for its own sake. Every one of the bugs mentioned above — the discarded
validation set, the shared-pickle collision, and (in a later chapter of this saga)
a transposed tensor axis and a forward-fill step that silently did nothing — was
already sitting in the original notebooks, undetected, because nobody had built the
scaffolding required to actually *notice*. Tests, tracking, and orchestration aren't
decoration on top of a working pipeline. They're how you find out whether the
pipeline was working in the first place.

**Next Steps**

With the foundation in place, our heroine turned her attention to a much stranger
question: was any of this pipeline's underlying *statistics* actually sound? That
turns out to be its own saga, currently running to at least two more installments.

**AI Use Statement:** Claude Code wrote both the code described in this post — the
package scaffold, the pydantic config validation, the MLflow/Prefect/DVC integration,
and the bug fixes named above — and this post's prose itself, across an extended
collaborative session with the author.
