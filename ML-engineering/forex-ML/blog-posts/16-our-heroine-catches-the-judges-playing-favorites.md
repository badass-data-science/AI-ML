# Our Heroine Catches the Judges Playing Favorites

Last chapter left one thread dangling on purpose. Of three new ideas fed to
the mule (the sturdier model this investigation had settled on), two made
things worse and one — teaching it to watch what the other currencies in the
stable were doing — actually nudged the win rate above fifty percent for the
first time in ages. Not enough to call it a discovery yet; it still hadn't
turned a real profit, and this project's whole track record this session was
one "promising" number after another failing to survive a second look. Before
chasing that thread any further, though, a more basic question got asked:
was the scoreboard itself trustworthy?

**Checking Whether the Race Was Called Fairly**

Every trade this entire investigation has ever backtested amounts to a bet on
which of two things would happen: price goes up, or price goes down. Under
the hood, the labeling scheme actually runs BOTH possibilities as independent
races every single time — a long bet and a short bet, each timed against its
own finish line. Whichever one crossed first became "the answer" for that
row. That part was solid, carefully built, and already tested.

What turned up on closer inspection was this: only the WINNING race's actual
result ever got kept. The loser's real outcome — how it would actually have
finished, in real time, at its own real price — got thrown away. That's fine
if a model always bet on the side that actually won. But roughly half the
time, by design, it doesn't — that's what "wrong" means. And when a model bet
on the LOSING side, the scoreboard had no honest number to give it. It quietly
handed back the winning side's result instead, as a stand-in, for every single
wrong-direction bet, all session long.

Picture a two-horse race where only the winner's finish time gets written
down. If you'd bet on the horse that lost, the record book doesn't actually
know how your horse did — it just hands you the winner's time and calls it
close enough. Most of the time it probably is close enough. But not always,
and there's no way to tell which without keeping both horses' real times.

That's exactly what got checked, and quantified, before touching a line of
code: out of over fourteen thousand real rows, the worst possible version of
this mistake — a losing bet that would ACTUALLY have won, had its own true
result been kept — never once happened. Reassuring. But the more ordinary
version of the mistake, the two races finishing at genuinely different times
with genuinely different results, happened often enough and by a large
enough margin on occasion to be worth fixing properly rather than waved off.

So both horses' real times are now kept, always, for every race. A model's
wrong-direction bet gets priced by what its own bet would actually have
done, not a borrowed number from whichever bet happened to win. Every test
this project has, plus two brand new ones built specifically to catch this
exact mistake if it ever crept back in, all pass. Re-running last chapter's
best-looking backtest through the corrected scoreboard moved the numbers by a
small, believable amount — worse in most spots, not dramatically so — exactly
what a real but modest fix should do. Nothing about the bigger picture
changed. Which was itself worth knowing: the fix was real, but it was never
the reason nothing here has turned a profit.

**The Last Horse Crosses the Line**

With the scoreboard now trustworthy, the one dangling thread from last
chapter finally got its proper hearing: the same five-way, non-overlapping
stretch of real market history used to debunk every other promising number
this session, run again on the "watch the other currencies" idea.

The picture across the five stretches was, again, wildly uneven — one
strongly good, one badly bad, one mildly bad, two more strongly good. Pooled
together, the win rate landed almost exactly on a coin flip, same as every
threshold tried. But something looked different this time: the total money
made, pooled across all five stretches, came out POSITIVE at every single
threshold tested. Not by accident of one lucky stretch dragging the others —
genuinely positive on net, even with a coin-flip win rate.

That's an unusual enough pattern to deserve its own honest test, not just the
usual one. A coin-flip win rate with money still coming out ahead means the
wins are bigger than the losses, not that there are more of them — a
completely different shape of "edge" than everything chased so far, and the
standard win-rate significance check doesn't even ask that question. So a
second, more appropriate test got run directly on the actual profit-and-loss
of every single trade, pooled across all five stretches — not just whether
each trade won or lost, but by how much.

It came back negative too. Not "obviously nothing," the way most of this
session's failures did — the best threshold got respectably close — but not
close enough to trust, either.

**Where Things Stand**

A tidy, honest chapter to close out a long thread. One real, previously
unnoticed flaw in how every single backtest all session long has scored a
wrong-direction bet — found, quantified, fixed, and confirmed not to be the
reason for any of this investigation's disappointing results. And the one
lead from last chapter that looked meaningfully different from everything
else tried — genuinely different in shape, not just in degree — given its
fairest possible hearing, with a test built specifically for the pattern it
showed, and still coming up short.

Two model families. Two different questions asked of them (which direction,
and by how much). Three separate rounds of giving the model more to look at.
A full audit of the very scoreboard used to judge all of it. None of it has
yet produced something this project can stand behind as a real, working edge
on this one pair, at this one timescale. That's not a small amount of
ground covered, and very little of it was wasted — every rejected idea
narrowed things down honestly instead of getting quietly buried. But it's a
natural point to ask a bigger question than "what feature should we try
next": is EUR/USD at hourly resolution the right race to keep entering at
all, or is it time to see whether a different pair, or a different
timescale, gives this whole approach more to work with.

**AI Use Statement:** Claude Code investigated and fixed the backtest-pricing
bug described in this post (auditing `simulate_trades`, quantifying the
mispricing against real data before touching any code, implementing the fix
across both the labeling and backtesting layers, and writing the regression
tests that pin it down), re-ran the cross-pair multi-window validation
through the corrected backtest, designed and ran the follow-up per-trade
significance test once the win-rate test didn't tell the whole story, and
wrote this post's prose. The author asked for the backtest to be
double-checked before trusting any more of its numbers, and set the
direction at each decision point that followed, but did not write any of the
code or run any of the analysis herself.
