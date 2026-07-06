# Our Heroine Learns Not to Cheat at Time Travel

Our heroine had a working, tracked, orchestrated pipeline now, which felt fantastic
for approximately one afternoon — right up until she remembered that none of that
matters if the model is quietly cheating. Not "sneaking answers on a test" cheating.
Something subtler and, frankly, more embarrassing: letting the model see the future
before asking it to predict the future, and then acting surprised when it's good at
its job.

This is the founding sin of applying ordinary machine-learning hygiene to time
series. Standard k-fold cross-validation shuffles your data before splitting it,
which is a wonderful idea for, say, photographs of cats, and an absolutely terrible
idea for anything indexed by time — because shuffling erases the one property that
actually matters: which rows happened *before* which other rows. Our heroine was not
about to let a forex model train on Tuesday and get quizzed on Monday.

**Rule One: Time Only Moves Forward**

`TimeSeriesSplitter`, the class doing the actual splitting, enforces strict
chronological order: every row in `train` happened before every row in `val`, which
happened before every row in `test`. No shuffling, ever. This sounds almost too
obvious to write down, right up until you inherit a codebase and have to go check,
personally, with your own eyes, that nobody quietly slipped a `.sample(frac=1)` in
there at 1 AM. (Nobody had. Our heroine checked anyway. This is called professionalism,
or possibly trust issues. The line is blurry.)

**Rule Two: Purge the Seams**

Here is the sneaky part. Even with a perfectly chronological split, the row sitting
*right at* the train/validation boundary can still cause trouble. A model's input
window reaches `n_back` bars into the past, and its label reaches `lookahead` bars
into the future — so a row near a boundary can have a window or a label that quietly
pokes across into the "wrong" side. This isn't leakage in the sense of the model
literally seeing future prices at inference time. It's something more like two
neighbors who share a very thin wall: the rows on either side of the seam are highly
autocorrelated, and that can optimistically bias whatever metric you compute right at
the join.

The fix, straight out of Marcos López de Prado's *purged k-fold cross-validation*: carve
out a gap of `max(n_back, lookahead)` bars on both sides of every split boundary and
throw those rows away entirely. A small tax on your row count, paid once, in exchange
for never having to wonder whether your validation score is a little too flattering.

**How Much History Does the Model Actually Need?**

`n_back` and `lookahead` — how many bars of history to look back, and how many bars
ahead to predict — were numbers carried over from the original notebooks with, as far
as anyone could tell, no empirical justification whatsoever. They were vibes. Good
vibes, possibly, but vibes.

Enter the autocorrelation function (ACF) and its shier cousin, the partial
autocorrelation function (PACF) — two classic tools for asking "how correlated is
this series with a lagged version of itself?" and "how correlated is it, once you've
already accounted for everything in between?" respectively. Compute these against a
pair's real target column, find the lag at which the correlation stops being
statistically distinguishable from zero, and you have a data-driven floor on how much
history actually carries *linear* signal. (A floor, not a ceiling — an LSTM can
exploit nonlinear structure an ACF plot is blind to. But if `n_back` is four times
larger than what the ACF suggests you need, that's worth a second look rather than a
shrug.)

Then our heroine ran into a classic trap: with the hundreds-to-thousands of bars
typical here, even a laughably small correlation — 0.02, functionally noise — can
still clear the bar of "statistically significant," because significance testing gets
easier, not harder, as your sample grows. So the diagnostic now also reports an
*effect size*: the actual magnitude of the correlation at the suggested cutoff, and a
second, "practical" cutoff based on a fixed correlation threshold that doesn't shrink
as the dataset grows. PACF gets its own version of both, since PACF — not ACF — is the
standard tool for spotting where an autoregressive process actually cuts off, and the
two can legitimately disagree.

**Is This Series Even Stationary?**

A related, equally uncomfortable question: does this feature's statistical behavior
stay put over time, or is it drifting out from under the model? Two classical tests,
ADF and KPSS, get run together here — deliberately, because they test *opposite* null
hypotheses. ADF's null is "this series has a unit root" (non-stationary); rejecting it
is evidence *for* stationarity. KPSS's null is "this series is stationary"; rejecting
*that* one is evidence *against* it. Run only one, and you inherit that one test's
blind spots. Run both, and either they agree (a strong signal) or they disagree — in
which case the honest answer is `inconclusive`, not a coin flip dressed up as
confidence.

And once again: a p-value alone isn't the whole story. With enough bars, ADF will
reject the unit-root null for almost any realistic financial series, even a highly
persistent one that behaves practically like a random walk over the horizons that
actually matter. So the diagnostic also reports `phi_hat` — the AR(1) coefficient,
pulled directly out of ADF's own regression rather than a second one fit
separately — converted into a half-life in bars, an actual interpretable unit. KPSS
doesn't have quite as clean a real-world unit to offer, but its raw statistic,
normalized against its own 5% critical value, still tells you *how far past the
line* you are, not just which side you landed on. A ratio of 3.0 and a ratio of 1.01
are both "non-stationary" by the p-value alone; only the ratio tells you they aren't
the same kind of non-stationary at all.

**The Moral, So Far**

None of this changes what the LSTM ultimately learns. All of it changes whether our
heroine can trust the number she gets back when she asks "how good is it?" — and, as
it turns out, trusting that number without a fight is exactly the next mistake
waiting for her.

**AI Use Statement:** Claude Code wrote both the code described in this post — the
purge-gap logic, the ACF/PACF effect-size diagnostics, and the ADF/KPSS stationarity
diagnostics — and this post's prose itself, across an extended collaborative session
with the author, who directed the statistical approach at each step.
