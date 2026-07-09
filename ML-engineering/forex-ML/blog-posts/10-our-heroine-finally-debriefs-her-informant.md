# Our Heroine Finally Debriefs Her Informant

Several chapters back, on the trading side of this story, our heroine recruited
three new informants for her investigation: an economic calendar, a snapshot of
retail positioning, and — the one this chapter is actually about — real swap
rates, pulled straight from OANDA and dropped into the database. At the time, she
was careful to say what that recruitment did and didn't mean: the data existed,
but nothing downstream had actually gone and talked to it yet. Triple-barrier
labeling's cost math ran on a placeholder constant, `0.0`, as if holding a
position overnight cost nothing at all.

It was time to actually debrief the informant.

**What the Informant Actually Says**

OANDA's `longRate`/`shortRate` fields — the numbers `SwapRateETL` had been
faithfully collecting all along — turned out to need translating before they were
useful. They're **annual** rates, expressed as decimals (`0.05` means 5% a year),
not the per-night figure the labeling code actually wanted. Plug the raw number in
directly and every trade gets charged roughly 365 times too much for a single
night's hold. The fix was one division (`/365`, a plain Actual/365 approximation —
OANDA doesn't publish enough detail about their exact day-count convention to do
better than that) and one sign flip: a negative rate means you're charged, so it
becomes a positive cost; a positive rate means you're credited, so it becomes a
negative one.

A detail worth sitting with, because it wasn't obvious going in: `long_rate` and
`short_rate` are not the same number wearing two hats. They're independently
signed. One real pair in this project's own test data has a *positive* long rate
and a *negative* short rate at the same time — meaning holding it long earns you a
little, and holding it short costs you a little, which is exactly backwards from
what you'd guess if you assumed one side was just the other side's mirror image.
Whatever informant gets asked, both answers have to be taken at face value, not
inferred from each other.

**A Convention That Turned Out to Already Be Handled**

Real FX brokers have a well-known quirk: swap gets charged triple on Wednesdays,
to cover the weekend a position sits through without anyone paying attention to it
day by day. Our heroine went looking for where she'd need to bolt that rule on —
and found she already didn't need to. The rollover-counting function built two
chapters ago doesn't compress the weekend into one Wednesday multiplier; it just
walks forward one calendar day at a time and counts every 5pm-New-York boundary a
trade actually sits through, Saturday and Sunday included. A position held across
a full weekend gets counted as crossing three boundaries — a different accounting
than OANDA's own, but the same total charge. No new code needed. Sometimes the
right amount of caution earlier pays for itself later.

**Two Bugs the Investigation Turned Up**

Neither of these was what the chapter set out to find, which is usually how the
better catches happen.

The first: a shared utility both this project and its trading-side sibling depend
on had been silently mis-converting timestamps for a specific shape of database
query — the kind nobody had happened to write yet, until this chapter's live rate
lookup needed exactly that shape. The bug had been sitting there the whole time,
harmless only because every existing query happened to avoid it by accident. It
surfaced during an ad hoc diagnostic check, not a scheduled test — the value came
back looking like January 1970 instead of this week, which is the kind of wrong
answer that's impossible to miss once you're actually looking at it.

The second was subtler: it lived in the gap between *when* a rate gets resolved
and *when* it gets logged. Fetching and labeling happen in one process; training
happens in a separate one, sometimes hours or days later. If training quietly
re-fetched "the current rate" instead of remembering what had actually been baked
into the labels it was training on, the number written to the experiment log
could end up describing a rate that had nothing to do with the data it was
sitting next to. The fix was to make the labeled data carry its own receipt —
save the exact rate used, alongside everything else, so nothing downstream has to
guess or re-derive it.

**"Do We Need to Delete and Refresh the Data?"**

This was the first question asked once the wiring was done, and it deserved a
real investigation rather than a guess. The short answer: no. The bug that got
fixed lived entirely on the *reading* side — the code path that writes a real
snapshot to the database never went anywhere near it. Every value already sitting
there was checked directly and came back sane.

The more interesting version of the question turned out to be "is anything
actually keeping this fresh going forward," and the first honest answer to that
was wrong. A quick process check for a scheduler came up empty — worth admitting
plainly, since it briefly looked like a real gap. Checking the actual deployment
system properly (not just guessing from what processes happened to be visible)
told a different story: a live, unpaused, correctly-scheduled job really was
running, and the sparse two-snapshot history wasn't a broken pipeline at all —
just a young one, exactly as many data points as you'd expect from one manual run
plus one real scheduled firing, with the next one not due yet. Getting the first
check wrong and then going back to verify properly, rather than letting a quick
guess stand in for an answer, is worth remembering as its own small lesson.

**"Do We Need Historical Swap Rate Data?"**

No — and this one has a cleaner answer than it might sound like. OANDA's API
doesn't expose historical rates at all; asking it "what was this rate in 2019" has
no answer to give. So the design that got built doesn't try: it fetches whatever
the *current* rate is and applies that single number as a flat constant across
the model's entire training history, however many years deep that history goes.
Not because point-in-time accuracy wouldn't be nicer, but because it isn't
available, and reconstructing it from public interest-rate data would be a real,
separate project of its own — not something to smuggle into a chapter about
wiring up a value that already exists.

**"Does That Retroactive Application Cause a Real Problem?"**

This is the question worth the most care, because "we're using a slightly wrong
number" is a very different claim from "this is a validity problem," and it's
tempting to conflate them.

Here's what actually determines the answer: how big is the swap-cost term
compared to the profit-take and stop-loss thresholds it gets subtracted before
comparing against? For the pair checked directly, a single night's real cost
comes out to a few thousandths of a percent — and since the maximum holding
period is a single day, most labeled trades cross at most one of those charges
before resolving. Against thresholds set at three-tenths of a percent, that's a
sliver — a couple percent of the barrier's width, not a meaningful fraction of
it.

Which means the retroactive-application question only bites in one narrow place:
trades whose outcome was already a photo finish, landing within that same sliver
of a threshold. For those, and only those, whether the swap charge used today's
rate or 2019's real rate could tip a label from a win to a timeout, or a timeout
to a loss. Everything else — any trade that cleared or missed a barrier by more
than a hair — never notices which rate got used, because the difference is too
small to matter next to how far the barrier was cleared or missed.

It's also worth being precise about what kind of wrongness this is, since not all
wrongness behaves the same way. This isn't noise that averages out — it's a
consistent lean, in one direction, for whatever historical stretch had genuinely
different real rates than today's. The years when real-world rates sat near zero
would have their labels nudged very slightly toward "the market had to work
harder to clear the same bar" than they should have been. That's a real
distortion, not a wash — it just happens to be a small one, confined to the
thin band right at the threshold, on top of thresholds (`profit_take_pct`,
`stop_loss_pct`, `max_holding_bars`) that are *already* flagged, in this
project's own configuration, as first guesses nobody has validated yet. A small,
well-understood bias sitting on top of a much bigger, already-acknowledged
uncertainty isn't nothing — but it isn't the thing to lose sleep over first,
either. Worth revisiting specifically if those thresholds ever get tuned down
tight enough that a few thousandths of a percent stops being a sliver and starts
being a real fraction of the bar.

One asymmetry worth naming on the way out: this same "use today's number"
approach behaves differently depending on which side of the project it's
standing in. Applied to *training* labels, it's smeared across a decade-plus of
history, drifting further from accurate the further back a label sits. Applied to
a *backtest's* held-out test window — which is, by construction, the most recent
slice of time available — today's rate is a much closer match to what was
actually true when those trades would have happened. Same mechanism, two
different amounts of honesty, depending on how far back it's being asked to
speak for.

**Where Things Stand**

The informant has been debriefed, translated correctly, and is now actually
listened to — by both the labeling math and the backtest's own cost accounting,
each charged the rate that actually applies to its own side of a trade. What
isn't pretended is that this makes the swap-cost figure historically accurate
going back over a decade; it doesn't, and can't, given what's actually available
to ask. What it does do is replace a placeholder that was silently wrong in every
single case with a real number that's only slightly wrong in a small, bounded set
of cases — and now that the scheduler is confirmed running, that number keeps
getting fresher on its own, with no repeat debriefing required.

**AI Use Statement:** Claude Code designed and implemented the swap-rate wiring
described in this post (the OANDA rate conversion, the InfluxDbTool timestamp
bug and its fix, the split/train process-boundary fix, and the direction-aware
backtest changes), investigated and answered the delete/refresh, historical-data,
and retroactive-application questions described above (including catching and
correcting its own initial wrong answer about whether a scheduler was running),
and wrote this post's prose itself. The author asked each of the questions this
post is structured around and made the scoping calls described (accepting a
current-rate-as-constant approach rather than commissioning historical rate
reconstruction).
