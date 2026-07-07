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
method gets.

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
cyclical features, and wrote this post's prose itself, across an extended
collaborative session with the author, who requested the tooling, asked the
questions that shaped its design, and reviewed each finding as it came in.
