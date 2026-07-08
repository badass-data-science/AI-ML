# Our Heroine Fires the Volatility Model

A couple of chapters back, our heroine was careful to draw a hard line: the
triple-barrier labeling scheme she'd just built — watch a trade forward,
call it a win, a loss, or a timeout, whichever barrier gets hit first — was
a *research tool*, not a policy. The model actually being trained still used
the old fixed-horizon target. Swapping it in for real, she said at the time,
was "a real, separate decision," deliberately left for later.

Later arrived. Over on the modeling side of this story, that separate
decision got made — triple-barrier labeling is now the only thing a model
gets trained on. Which meant this trading-side code, built against a world
where models came in two flavors sharing a pair, had some catching up to do.

## The Tag That Used to Mean Two Things

Every model version registered here has always carried a `column_y` tag,
and it used to earn its keep by telling apart two genuinely different
models that could both be sitting under the same instrument and
granularity: a directional model and a volatility-forecasting one, needed
together for the position-sizing trick described below. Filtering on that
tag was the whole reason a lookup for "the model for EUR/USD" didn't
accidentally hand back whichever of the two happened to be trained most
recently.

There's only one trainable target now, so `column_y` always reads
`"triple_barrier"` — which sounds like the tag lost its job. It didn't,
quite. Older model versions, registered before this migration, are still
sitting in the registry tagged the old way, or not tagged at all. Filtering
on `column_y="triple_barrier"` is what keeps one of those from getting
picked up by a lookup that just wants "the latest version for this pair,"
regardless of which era it came from. Same tag, same filter, a different
reason for needing it.

## Retiring a Second Hire

The bigger change is the one that used to require hiring a second model
entirely. Sizing a position down when volatility looked likely to spike
meant training a whole separate network to forecast an ordinal
low/medium/high volatility class, registering it alongside the directional
model, downloading *its* predictions too, and — because the two models'
test sets had to line up row-for-row by timestamp to be combined at all —
checking that alignment explicitly, with a dedicated regression test to
catch it if it ever broke.

All of that is gone now. The modeling side's own investigation found
something the old approach was working around unnecessarily: volatility
has real memory, verified with a shuffle test months ago showing genuine
day-to-day persistence beyond the daily clock cycle. If the recent past
already predicts the near future for volatility specifically, there's
nothing to gain by training a second model to reconstruct that fact — you
can read it straight off the tape. A single new field, `test_realized_volatility`,
now rides along in the same predictions file the directional model already
produces, and position size scales against it directly: smaller when
recent volatility is running hot, capped at full size when it isn't. One
model, one predictions file, nothing left to keep aligned by hand.

## A Free Upgrade to the Clock

Charging swap cost, or deciding whether to flatten a position ahead of
rollover, both need to know when a trade actually ends. The old code
guessed: every trade was assumed to last exactly however many bars the
model's fixed lookahead specified, whether it actually took that long or
not. Triple-barrier labels don't work that way — a trade resolves the
moment a barrier gets hit, which could be one bar in or most of a day in.
That real, per-trade resolution time was already being recorded on the
modeling side for an unrelated reason; all this side had to do was start
reading it. Exit timestamps are now computed from each row's own actual
holding period instead of a single number assumed to apply to every trade
alike — a more honest rollover count, and one that came essentially free.

## The Numbers Stayed Honest

One small rename carried a real point. The backtest's core P&L function
used to take a parameter called `pd_lead_pct` — a name describing exactly
one target that no longer exists. It's `raw_return_pct` now, and the "raw"
matters: it's still deliberately *not* the net-of-cost figure the labeling
step already computes internally. This backtest charges its own spread and
its own swap cost on every trade; handing it a number that's already net of
those same costs would charge every trade twice for the same thing. Small
detail, easy to get backwards, and exactly the kind of thing this project
has cared about since its very first ledger.

## Where Things Stand

Three test fixtures that used to train a directional model, a volatility
model, and both together, collapsed into one — there's only one thing to
train now, so there's only one thing to fake for a test. What's still
missing is the obvious next step: an actual trained triple-barrier model to
point this at for real, which doesn't exist yet on the modeling side
either. This chapter made the trading code honest about what it's now
reading. Whether what it reads turns out to be worth trading is still an
open question, and — same as it's always been in this story — an honest
"not yet" beats a confident guess.

---

**AI Use Statement**: Claude Code implemented the changes described in this
post (the `column_y` filter update, removing the second-model volatility
lookup in favor of `position_size_from_realized_volatility`, the
`test_exit_bar_offset`-based exit-timestamp fix, and the `raw_return_pct`
rename) and drafted this post's prose, across a session where the author
had already decided, on the modeling side, to make triple-barrier the only
trainable target — this post covers the trading-side code catching up to
that decision.
