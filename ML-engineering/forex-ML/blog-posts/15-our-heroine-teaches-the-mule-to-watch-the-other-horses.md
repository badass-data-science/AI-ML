# Our Heroine Teaches the Mule to Watch the Other Horses

Last chapter left the mule (the sturdier, more trustworthy model this
investigation had settled on) in an honest but frustrating spot: it had
proven, twice over, that it wouldn't fall apart the way the old thoroughbred
did — but it still hadn't won any actual money. Asking it to guess direction
instead of magnitude hadn't fixed that. Asking it to guess magnitude instead
of direction hadn't fixed that either. So the question left on the table was
the plainest one left to ask: was the mule simply not being shown enough to
work with?

Up to this point, every version of every model had been handed the exact same
kind of information — a trailing window of this one currency pair's own
recent price wobbles, nothing more. No sense of the broader trend running
underneath those wobbles, no sense of whether the market was calm or agitated
lately, none of the classic "is this overbought or oversold" instincts a human
trader reaches for by habit, and no idea whatsoever what any *other* currency
was doing at the same time. Three attempts to fix that, in order.

**A Longer Memory for the Trend**

First, the cheapest fix: alongside the short-term wobble-tracking the model
already had, give it a much longer look-back — several trading days' worth —
so it could tell the difference between "a brief blip" and "the market's
actual current mood." Paired with that, a simple ratio comparing short-term
choppiness to that longer-term baseline, so the model could tell an expanding,
nervous market from a calm, settled one. Cheap to add, easy to justify. It
made no real difference. If anything, the very next check — a real
money-weighted test — came back worse than before.

**Some Classic Instincts**

Next: the kind of signals any human trader picks up early — is the recent
trend accelerating or fading, is this move statistically unusual for the
pair's own recent behavior, does the balance of recent up-moves versus
down-moves suggest it's overdue for a turn. All three were rebuilt from
scratch rather than borrowed off the shelf, because the textbook versions are
built on raw price levels, and this project made a firm decision a long time
ago never to feed the model raw, wandering price levels directly — only
already-settled, self-relative measures. Same story as the first attempt:
tested honestly, and it didn't help. Accuracy nudged up by a hair; the actual
money-weighted test got worse, not better.

**Watching the Other Horses**

The third idea was the most different in kind, not just in degree: stop
looking only at this one pair, and start checking what the other currencies
this project already tracks were doing at the very same moment. Every pair in
this stable is priced against the same currency, so averaging their moves —
flipping the sign for the ones quoted the opposite way round — produces a
rough read on whether that shared currency was broadly strengthening or
weakening across the board, independent of any single pair's own particular
story.

This one actually moved something. For the first time since the very original
feature set, the win rate climbed back above fifty percent — not at just one
lucky threshold, but at three separate confidence levels — and the best
result landed far closer to break-even than anything the last two attempts
had managed. It still didn't cross into genuine profit; the cost of actually
placing all those trades still ate more than the edge produced. But of three
honest attempts to give the model more to look at, this was the only one that
left a real, if modest, mark rather than making things worse.

**Where Things Stand**

Twelve new, real, fully tested features went in this chapter, spread across
three separate ideas, and by the account that actually matters — real money,
after real costs — none of them turned a profit. That's a plain, unflattering
sentence to have to write after this much work. But it isn't the same as
"nothing was learned." Watching the other horses is the first idea in a long
while that nudged the needle in the right direction instead of the wrong one,
and it's a different *kind* of idea than everything tried before it — not a
tweak to how the mule reads this one pair's own history, but a genuinely new
source of information entirely. That's worth remembering, and worth returning
to, even though it isn't yet a finish line.

**AI Use Statement:** Claude Code ran this entire investigation described in
this post — designing and implementing all three feature additions
(multi-timeframe/volatility-regime, momentum-oscillator analogs, and the
cross-pair strength signal, including the underlying pipeline change needed to
pull other pairs' data), writing their tests, regenerating the underlying
data, running every backtest check, and writing this post's prose. The author
set the direction at each decision point (choosing "richer features" as the
next avenue over other options, approving the specific feature plan before
implementation, and deciding to push through all three sub-ideas before
stopping to take stock) but did not write any of the code or run any of the
analysis herself.
