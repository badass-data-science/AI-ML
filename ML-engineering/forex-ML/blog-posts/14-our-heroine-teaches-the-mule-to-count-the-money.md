# Our Heroine Teaches the Mule to Count the Money

Last chapter ended with a trade our heroine felt good about, on balance: she'd
swapped an unpredictable, occasionally brilliant thoroughbred (an LSTM prone to
mysteriously giving up and guessing one answer forever) for a plodding, reliable
mule (a gradient-boosted tree). The mule had never once collapsed, never once
needed rescuing mid-race. It just hadn't won any money yet. Three follow-up
questions were queued up: could the thoroughbred be fixed with a shorter
memory, did the mule's advantages actually travel beyond the one racetrack
they'd been tested on, and — the big one — was asking either animal "which
direction" the wrong question, when what actually paid the bills was "by how
much"?

**A Shorter Memory Didn't Help**

First, the easy one. If two hundred hours of lookback was somehow confusing the
thoroughbred, maybe fifty would settle it down. It did fix one thing: the
model stopped diverging into nonsense mid-training entirely. But it made the
main problem — collapsing to guessing one answer over and over — happen
*every single time* instead of less than half the time, and the accuracy that
survived was worse than just guessing the majority. A shorter memory didn't
calm the thoroughbred down. It just gave it less to work with before it gave
up. Discarded.

**Taking the Mule on the Road**

Next: was the mule's good behavior a fluke of the one currency pair it had
been tested on? She sent both animals out to three new tracks — GBP/USD,
USD/JPY, AUD/USD — same hour-by-hour data, same race conditions. The verdict
held up everywhere she looked. The mule stayed calm and consistent on every
single pair, and feeding it the *entire* recent window of data instead of just
the most recent tick made it noticeably sharper everywhere, not just at home.
The thoroughbred, meanwhile, took her spot-checks on the new tracks about as
well as expected — it fell apart there too. Whatever's wrong with the
thoroughbred, and whatever's right with the mule, isn't a EUR/USD-specific
quirk. It travels.

**Teaching the Mule to Count the Money**

Now the real question. Every model so far had been asked the same kind of
question: is this about to go up, down, or nowhere? That's a fine question if
you only care about direction. But the actual goal was never "guess the
direction correctly" — it was "make money," and a model can guess the
direction right every time while still losing, if it's confidently right about
small moves and confidently wrong about big ones, or the reverse. So she tried
something different: instead of asking the mule to pick a lane, she asked it
to guess the actual size of the payoff — in effect, to count the money
directly — and only place a bet when its guess cleared some minimum size.

The first result out of this was, frankly, thrilling. On one stretch of
market history, betting only when the mule's predicted payoff was large
enough produced a win rate over fifty-two percent and a net result, after real
trading costs, of plus twenty percent. The best single number the entire
investigation had produced, by a wide margin.

She'd been burned by a good-looking number exactly once before, and the fix
for that was already sitting in the toolbox: don't trust one stretch of
history, trust five. Same test, same threshold, run across five separate,
non-overlapping windows of real market data instead of just the one. The
verdict was almost identical to the last time this trick was tried — the
promising window turned out to be a single lucky lap, not a real pattern.
Pooled across all five, the win rate settled right back to a coin flip at
every threshold tried, and the overall money made across all five windows
combined was negative, not positive. One truly excellent window, one truly
bad one, and three unremarkable ones in between, averaging out to nothing.

**Where Things Stand**

A tidier chapter than the last one, if a less triumphant one. One idea (shorter
memory) got tried and cleanly rejected. One idea (does this generalize) got
tried and cleanly confirmed — the good and bad behaviors of both animals
follow them wherever they race. And the one idea with the most riding on it —
teach the mule to count the money instead of just pick a lane — produced this
investigation's single best number and then, on the very next honest look,
gave it right back.

The mule is still the more trustworthy animal. It has now proven that twice,
on four different tracks, under two different kinds of questions. What it
still hasn't proven, under any question asked of it so far, is that it knows
how to actually win. That's a different, harder problem than the one this
project spent its last few chapters solving — and it's the one still sitting
on the table.

**AI Use Statement:** Claude Code ran this entire investigation described in
this post — the shortened-lookback experiment, the multi-pair survey across
three additional currency pairs and two feature variants, setting up and
evaluating the return-regression approach, the initial single-window backtest,
and the five-fold multi-window validation that debunked it — and wrote this
post's prose. The author set the direction at each decision point (ordering
the generalization check before further tuning, proposing the shift from
classification to a directly money-shaped target, insisting on multi-window
validation before trusting the promising regression result) but did not write
any of the code or run any of the analysis herself.
