# Our Heroine Interrogates Her Own Good News

There is a particular flavor of joy that arrives when a model reports a test accuracy
of, say, 45% on a three-class problem, and a particular, much less pleasant flavor of
realization that arrives about four seconds later, when our heroine remembers that a
coin flip between three classes gets you 33% for free. The gap between those two
numbers is where the actual question lives — and our heroine, having recently sworn
off taking things at face value, decided to make a habit of asking it.

**Baselines: The Bar You Actually Have to Clear**

Every training run now computes two trivial baselines alongside the LSTM's own test
accuracy: a **majority-class baseline** (always predict whatever class was most
common in training) and a **persistence baseline** (predict that this period repeats
whatever the previous period actually did — a classic, annoyingly strong baseline in
financial time series, where "nothing changed" is a surprisingly good guess).
Neither baseline touches the model or the input features at all. They're logged
right next to `test_accuracy` in MLflow for one reason: so that number is never read
in a vacuum. A model beating 33% isn't news. A model beating a persistence baseline
that already sits at 40% — that's a claim worth examining.

**Regime Drift: Watching for the Ground Shifting**

Training data gets bucketed into three roughly even classes by construction — the
thresholds are literally computed from training-set quantiles. Validation and test
data get no such courtesy. If the market's volatility regime shifted between the
training period and the test period, the test set's class balance can skew hard in
one direction, and a plain accuracy number won't tell you that happened. So every
split now logs its actual class balance per class, cheaply, right alongside
everything else.

This one came with a legitimate question attached, which our heroine is glad someone
asked out loud rather than assumed away: does logging the test set's class balance
leak information into the pipeline? The answer is no, and it's worth being precise
about *why*: `class_balance()` only reads the already-materialized `y` array and
writes a metric. It never touches `X`, never feeds back into the train-only threshold
computation, and is never passed to `.fit()` or `.evaluate()`. The test set enters the
model exactly once, at evaluation time, regardless of this logging. There is a
different, more human risk worth naming honestly, though: seeing "test period skewed
toward class 2" sitting right next to a disappointing accuracy number creates a
temptation to go adjust thresholds and try again. *That* would be leakage — introduced
by the person, not the code. The diagnostic is meant to explain a result you already
have, not to be iterated against.

**Multiple Comparisons: When Fourteen Coin Flips Start Looking Like a Pattern**

With fourteen `(instrument, granularity)` pairs, each getting its own "does the LSTM
beat the baseline?" hypothesis test, simple probability guarantees that *some* pair
will look significant purely by chance, even if not one of them has any real signal.
This is the exact problem a Benjamini-Hochberg false-discovery-rate correction exists
to solve — but only if you feed it every hypothesis you actually tested. Underfeed it,
and you get a very official-looking number that quietly means nothing.

McNemar's test (the correct paired comparison for two classifiers scored on the same
rows — it only looks at rows where the two models disagreed) runs per pair, and BH-FDR
corrects across all of them together. But here's where it got genuinely interesting:
our heroine builds this on a single, somewhat weary local GPU, with no cloud compute,
which means pairs get trained one at a time, over weeks, as data accumulates — and
architecture search on a given pair (more layers? fewer? different activation?
different learning rate entirely?) is expected to happen *often*.

That raises an uncomfortable question: if you retrain the same pair five times with
five different architectures and only report whichever one won, have you actually run
one hypothesis test, or five? The honest answer is five — picking the best of several
attempts and reporting only that one's p-value is exactly the kind of
researcher-degrees-of-freedom problem this correction exists to catch, and it gets
*worse* the more often architecture search happens, not better. So the grouping logic
now hashes the *entire* logged training configuration — not just layer count, every
parameter — to decide whether two runs on the same pair are "the same experiment,
rerun with more data" (which correctly collapses to the latest run) or "a genuinely
different configuration" (which correctly earns its own slot in the correction pool).
Whole-params equality, not a hand-picked subset someone thought to name — because the
whole point is catching the change you *didn't* think to name.

**Rolling Cross-Validation: Distrusting a Single Lucky Slice**

And finally, the deepest cut of professional paranoia: even a single, honestly
computed, properly-corrected train/val/test split is still just one sample from one
slice of history. A good result could reflect a genuinely good model, or it could
reflect an unusually forgiving three months. Rolling (walk-forward) cross-validation
addresses this the only way that actually works for time series: walk a train/val/test
window forward through the timeline, retrain fresh at every step, and report the
*distribution* of results across folds rather than a single flattering number.

Two flavors are supported. A **sliding** window keeps training-set size fixed and
lets it march forward with the fold — more honest about regime change, since stale
history ages out, at the cost of never using all the data you actually have. An
**expanding** window anchors at the very first bar and grows every fold — uses
everything available, at the cost of assuming last decade's forex market still
behaves like this decade's. Neither is universally correct; both are now one flag
away from each other.

Crucially, this is a *diagnostic*, not a deployment strategy — each fold logs to its
own, separate MLflow experiment, tagged with its fold index, and is deliberately never
registered as a model. That separation matters more than it sounds: without it, five
rolling-CV folds of the *same* configuration would either get silently swallowed by
the "most recent run wins" logic above, or would flood the multiple-comparisons pool
with runs that were never meant to be independent hypotheses in the first place. Good
fences make good statistics.

**The Moral, Continued**

None of this is our heroine assuming the worst about her own model out of spite. It's
the opposite, really — it's what taking the model *seriously* actually requires. A
number you haven't tried to disprove isn't a result. It's a rumor.

**AI Use Statement:** Claude Code wrote both the code described in this post — the
baselines, the regime-drift reporting, the multiple-comparisons correction (including
the whole-configuration grouping), and the rolling cross-validation diagnostic — and
this post's prose itself, across an extended collaborative session with the author,
who directed each statistical decision along the way.
