# Our Heroine Hires a Cheaper Detective

Ten columns now feed the model, each one a 200-bar window, each epoch costing
roughly thirteen minutes of a single GPU's undivided attention. Our heroine,
having just watched that same GPU nearly max itself out over a *smaller*
feature set a few chapters back, found herself facing an increasingly
familiar question: if she ever needs to add a candidate column, or cut one
because memory got tight, how would she know which one actually matters —
without paying for a full training run just to find out?

The honest answer, she realized, was that she didn't have a good one. Every
feature decision so far had been justified by a specific, hand-built
investigation — ACF here, a shuffle test there — each one useful, none of
them repeatable as a standing tool. So this week's project wasn't a new
feature. It was a cheaper way to ask the same question over and over again,
on demand, in the minutes it takes to make coffee rather than the hours it
takes to train a model.

**Four Detectives, Cheapest First**

The idea behind all four techniques below is the same: approximate "does
this column help predict the target" using *linear* methods, which run in
seconds on a CPU instead of minutes-per-epoch on a GPU. Linear approximations
can't see everything an LSTM can — they're a floor on what's worth a real
training run, not a ceiling on what could matter — but a floor built in
minutes is still a genuinely useful thing to stand on before spending hours.

*Cross-correlation* is the simplest: for each candidate column, correlate its
value some number of bars in the past against the target right now, across a
whole range of lags, and see where the correlation peaks. It's the exact
same math behind the ACF/PACF checks from a few chapters back, just pointed
at two different series instead of one series against itself.

*Granger causality* asks a sharper question: does a candidate's own history
improve a linear forecast of the target *beyond what the target's own
history already provides*? This is worth being precise about, because the
name causes more confusion than almost any other term in applied
statistics — "Granger causality" means "this column's past carries additional
linear predictive information," full stop. It says nothing about mechanism,
and it has a real assumption baked in that's easy to skip past: the series
being tested need to be reasonably stationary, or the test can report
significance that isn't really there. This tool checks that assumption
explicitly, rather than trusting a p-value blind.

*Multivariate autoregression*, or VAR, is Granger causality's more careful
sibling. Testing each candidate one at a time against the target, in
isolation, has a real blind spot: if several candidates are correlated with
*each other* — and four trading-session flags built from the same underlying
clock are exactly that — a pairwise test can call all four "significant"
even when they're substantially saying the same thing. A VAR models
everything jointly instead, controlling for every other candidate at once,
and comes with a genuinely useful side benefit: forecast-error variance
decomposition, which reports what fraction of the target's own forecast
uncertainty each candidate accounts for. It's the closest thing a purely
linear method has to an actual feature-importance ranking.

*Lasso*, finally, is the one built specifically to answer "what could I
drop." Regress the target against every candidate at a whole range of lags,
with an L1 penalty that pushes unhelpful coefficients toward zero rather than
just shrinking them a little. A candidate whose entire lagged block gets
zeroed out is about as direct an automated "you don't need this" as a linear
method gets. One detail worth a sentence of its own: picking *how hard* to
penalize is itself tuned by cross-validation, and the ordinary flavor of
cross-validation randomly shuffles data into folds — which, for a time
series, means letting next Tuesday quietly leak into the fold used to
evaluate last Monday. A specific, chronology-respecting variant (each fold
strictly later in time than the ones before it) stands in instead, for
exactly the same reason nothing else in this pipeline shuffles.

**Running It For Real, on EUR/USD Hourly Candles**

All four detectives were pointed at the same case file: the ten columns
currently feeding the model, real production EUR/USD data at the one-hour
granularity, evaluated against the actual `volatility_lead` target. Nothing
below is a synthetic demonstration — every number came out of roughly
seventy-two thousand real hourly bars.

Two columns came back unambiguously useful, agreeing across every single
method: `volatility` and `diff_volume`. Both showed a strong same-bar
correlation with the target, both survived the properly-multivariate VAR
test even after controlling for every other column at once, and Lasso kept
real, non-trivial coefficients for both. When four different techniques,
built on four different sets of assumptions, all point the same direction,
that's about as close to a confident answer as a purely linear method gets.

One column came back unambiguously weak: raw `return`. Its correlation with
future volatility was almost nothing — 0.016, barely a whisper above pure
noise. Lasso's regularization crushed its coefficient down to essentially
zero. And in a detail worth sitting with for a second: the simple, one-
column-at-a-time Granger test called `return` statistically significant,
while the properly multivariate VAR test — controlling for the other,
correlated columns already in the room — did not. That disagreement is
exactly the failure mode VAR exists to catch, and it's the same lesson this
whole project has run into before in a different costume: with seventy-two
thousand rows behind it, almost anything can clear the bar of "technically
detectable," which isn't the same question as "actually useful."

The trading-session flags told a genuinely nuanced story rather than a clean
one. `is_new_york_session` came back robustly important by the most
demanding test available — significant in the multivariate model, with the
single largest legitimate share of the target's forecast-error variance.
The other three session flags, and the two cyclical time-of-day features,
told a messier story that took real digging to sort out — recounted in
full below, because the reason turned out to matter more than the result
itself.

**A Detail That Almost Wasn't One**

Building the VAR piece surfaced something genuinely interesting, in the
specific way that a tool built with enough care tends to catch its own
edge cases rather than silently reporting nonsense.

Running the full toolkit against the real data, `day_cos` — one of the two
cyclical time-of-day features — came back with the single *largest* variance
share of any candidate in the forecast-error decomposition, and simultaneously
came back "not statistically significant" in the VAR's own causality test.
Those two results flatly contradict each other, and our heroine's working
rule by this point in the story is that a contradiction like that is a
signal to investigate, not a number to quote.

The explanation turned out to be genuinely elegant, once found: a sine wave
built from a perfectly regular clock has an exact mathematical property that
an ordinary noisy signal doesn't. Shift a fixed-period sine wave back by any
number of steps, and the result is always exactly expressible as a
combination of the *original*, unshifted sine and cosine — a basic
trigonometric identity, not an approximation. Feed eleven lagged copies of
`day_sin` and `day_cos` into a linear model that estimates all of them at
once, and it's staring at twenty-two columns that mathematically carry only
two dimensions' worth of actual information. Checked directly against the
real data: a rank of two, out of twenty-two — and a matrix so numerically
ill-behaved that its condition number came back over a trillion. No wonder
the significance test choked.

The fix wasn't to throw the finding away — it was to make sure the tool
itself would catch it automatically, for whoever runs this next, on whatever
candidate columns come up next. The VAR report now checks its own lagged
design matrix for exactly this kind of structural rank collapse and prints
an explicit warning when it finds one, so a future "this doesn't look
significant" is never quietly mistaken for "this doesn't matter" without at
least a chance to notice why.

That left the original question about `day_sin`/`day_cos` still open, though
— the VAR verdict for that pair specifically couldn't be trusted either way,
which isn't the same as an answer. Lasso, as it happens, doesn't have this
particular blind spot the way VAR's significance test does — faced with
redundant, collinear inputs, it just picks a representative and shrinks the
rest, rather than choking numerically. And Lasso's verdict was unambiguous:
every single lag of both `day_sin` and `day_cos`, zeroed out completely.
Once `volatility`, `diff_volume`, and the session flags are already doing
their job, the cyclical clock encoding doesn't appear to be pulling
additional linear weight of its own for *this specific target*. That's a
genuinely different question from "does it carry information the other
features lack" — an earlier chapter already showed it does, in the form of
two full hours of the day that no session flag ever touches — and both
answers can be true at once: real, distinct information that this particular
target doesn't happen to need.

**Caveats, Because Every One of These Tools Has a Way of Lying Convincingly**

None of the above should be read as more certain than it is, and two
different kinds of caveats apply — one about the tools in general, one about
this specific run.

About the tools themselves: every technique here is a *linear* approximation.
An LSTM can combine features nonlinearly, across time, in ways none of these
four methods can see — a column that looks useless to a straight-line
correlation could still matter to a network that learns to combine it with
three other columns in a curve no simple regression could express. This is
a floor on what's worth a training run, not a ceiling on what could possibly
help. Granger causality's name overpromises what it measures — "improves a
linear forecast" is the honest translation, not "causes." Granger's validity
also assumes reasonably stationary inputs, which is why every candidate's
stationarity verdict gets checked and reported alongside its p-value rather
than assumed. And with tens of thousands of rows behind every test here,
statistical significance gets easy to achieve for almost anything — the
`return`/VAR story above is exactly what it looks like when that gap between
"detectable" and "meaningful" actually shows up in a real result, not just a
hypothetical warning in a docstring.

About this specific run: every number above describes **one currency pair,
EUR/USD, at one granularity, hourly candles** — not forex broadly, not other
pairs, and not other timeframes, all of which could plausibly tell a
different story given how differently trading sessions and liquidity behave
across pairs. The VAR/FEVD verdicts for the entire day-and-session block
(`day_sin`, `day_cos`, and all four session flags) carry the rank-deficiency
caveat discovered above, to varying degrees — `is_new_york_session`'s result
looks the most isolated from that specific problem and the most trustworthy
of the six, but none of them should be read with the same confidence as
`volatility` or `diff_volume`'s much cleaner, unanimous verdicts. And
finally: this is one screening pass on one target column, meant to inform a
decision, not replace the training run that actually tests it.

**A Tool for Next Time, Not Just This Time**

None of this replaces an actual LSTM training run when the LSTM is what
ultimately matters. What it replaces is guessing — or worse, training blind
and finding out thirteen minutes per epoch too late that a column was never
going to help. The next time a candidate feature shows up, or GPU memory
gets tight enough that something needs to go, this is now a five-minute
question instead of a several-hour one.

**AI Use Statement:** Claude Code designed and implemented all four techniques
described in this post (cross-correlation, Granger causality, VAR with
block-exogeneity tests and forecast-error variance decomposition, and Lasso
lagged regression), discovered and fixed the rank-deficiency issue with
cyclical features, ran the analysis against real EUR/USD H1 data, interpreted
the results described above, and wrote this post's prose itself, across an
extended collaborative session with the author, who requested the tooling,
asked the questions that shaped its design, and reviewed each finding as it
came in.
