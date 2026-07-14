# Our Heroine Retires the Track, Not the Stable

Last chapter closed on a question rather than an answer: after auditing the
very scoreboard this whole investigation had been trusting, and finding one
real mistake in it worth fixing, was it time to try a different pair, or a
different timescale, before concluding there was nothing left to find? This
chapter tries the second option properly, then answers the bigger question
underneath it honestly.

**Trying a Different Track**

The idea was simple enough: maybe hourly data is just a noisier racetrack
than it needs to be. Widen the bars — four hours at a time instead of one —
and the theory goes that real signal has more room to show itself over the
noise of ordinary short-term wobbling.

Getting there took an unplanned detour first. This four-hour data didn't
exist yet in this project's own pipeline at all, so building it meant a trip
over to the sibling project responsible for actually collecting prices, and
along the way, a second real bug turned up — a much older one than anything
found this chapter. The system responsible for patching gaps in the data
(weekends, holidays, the ordinary outages of any live feed) had been quietly
assuming that every timekeeping boundary was a fixed number of clock-seconds
apart, forever. That assumption holds for hourly data. It does not hold for
four-hour data, because those particular bars are anchored to a local time
of day that itself shifts by an hour, twice a year, whenever clocks change.
The gap-patcher had been silently drifting out of sync with reality every
single time that happened — for years, on data already in production, not
just the new stuff. Confirmed, fixed, and — a nice touch — the very first
day it ever broke lined up exactly with a real clock-change date from over a
decade ago. Found because someone finally looked closely enough at a new
kind of data to notice the pattern.

With the track actually built and the timekeeping trustworthy, the horse ran
the exact same race as before: the honest, no-extra-features baseline, then
three separate rounds of giving it more to look at — a longer memory, some
classic momentum instincts, and a peek at what the other currencies were
doing. The most interesting single number to come out of any of it was a
net result close enough to exactly zero that it wasn't worth taking
seriously — nowhere near as convincing as the best number the *previous*
track had produced, and that one had already failed its own honest retest.
A different track, run the same way, with the same outcome.

**Reading the Full Scoreboard**

Here is the tally, once and for all, rather than scattered across a dozen
earlier chapters: two entirely different kinds of animal tried (an
occasionally-brilliant, occasionally-erratic thoroughbred, and a sturdier,
more consistent mule). Two entirely different questions asked of each one
(which way is this going, and separately, by how much). Four separate
rounds of richer information handed over, on two different racetracks. Extra
tracks checked briefly for good measure, just to see if the pattern held up
elsewhere — it did, in both directions: the thoroughbred was unreliable
everywhere it was tried, and the mule was steady everywhere it was tried. Every
single promising-looking number that surfaced anywhere in that search got the
exact same treatment: run it back across five separate, non-overlapping
stretches of real history before believing a word of it. None survived. Not
one.

And the scoreboard keeping track of all of it got checked twice, independently,
specifically *because* of how much weight was being put on its verdict — and
the one real mistake it turned up, once fixed, didn't change the final answer
even slightly.

That's not a small effort coming up empty by accident. That's a genuinely wide
search, conducted carefully, landing on the same honest answer from every
angle: there is no edge here that this project can responsibly stand behind.

**What This Doesn't Mean**

It would be a much bigger, much shakier claim to say no edge exists anywhere
in trading currencies, full stop — and that claim isn't being made. Whole
categories of information were never brought to this race at all: what
serious institutional order flow actually looks like moment to moment, real
economic news as it breaks (blocked, this whole time, by a data provider's
paywall rather than any fault of the approach), what other traders'
positions actually look like (blocked entirely, after the exchange quietly
retired the one feed that would have shown it). Nor was a fundamentally
different kind of race ever tried — a training process built from the start
to care directly about money made rather than a proxy for it. And this whole
search stayed on the biggest, most closely-watched, most competitively-traded
currency pair there is, precisely the sort of racetrack where every
easy-to-find edge has already been found and priced away by someone bigger,
faster, and better-resourced than a single independent project.

**What Actually Got Built**

None of which makes the last several months of work a loss. What comes out
of this isn't a trading strategy — it was never guaranteed to be one — but it
is a genuinely solid stable: a labeling scheme that mimics how a real trade
actually resolves rather than a fixed, arbitrary lookahead; a way of slicing
historical data for testing that never lets a model peek at its own future;
a full toolkit for asking, honestly, "is this result real or did I just get
lucky"; and a bookkeeping system for turning predictions into money, audited
twice over and trusted precisely because it was never assumed to be correct
in the first place. That infrastructure doesn't care what track it races
on next.

**Where Things Stand**

This particular pursuit — this pair, at these timescales, with the
information available so far — is closed, not abandoned mid-stride. The
next chapter starts somewhere genuinely different: not a new feature on the
same crowded track, but a new track entirely, one with less traffic on it.
Whether that goes any better is an honestly open question, which is exactly
how a question like that should stay until it's actually been asked.

**AI Use Statement:** Claude Code built and ran the H4 granularity comparison
described in this post (including finding, diagnosing, and fixing the
DST-related data-collection bug along the way, confirmed against real
multi-year history before and after the fix), ran the same three-round
feature-richness progression used on the original timescale, and wrote this
post's prose, including the closing summary of the full investigation. The
author asked directly, after seeing the H4 results, whether the forecasting
approach itself was the problem given a now-twice-audited backtest — a
question that framed this post's entire second half — and chose to close
this line of investigation and open the next one, but did not write any of
the code or run any of the analysis herself.
