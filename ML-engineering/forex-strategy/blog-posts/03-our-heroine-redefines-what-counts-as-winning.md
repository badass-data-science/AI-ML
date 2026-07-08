# Our Heroine Redefines What Counts as Winning

Somewhere in the middle of building an honest P&L ledger, our heroine noticed
something a little embarrassing about how her models had been *labeled* all along.
The target column they'd been trained on — "the percent price change exactly four
bars from now" — is a strange, slightly arbitrary way to define a trade. Real trades
don't wait around for a buzzer to go off at a fixed number of bars. A real trader
gets out when the price hits a profit target, or when it hits a stop-loss, or —
if neither happens — eventually gives up and closes the position anyway. "What
happened exactly four hours after I opened this" isn't really any of those three
things; it's just whatever number happened to land at a convenient, fixed distance
away.

This turns out to be a well-known problem with a well-known fix, called the
**triple-barrier method** (from Marcos López de Prado's *Advances in Financial
Machine Learning* — a book this project had already borrowed one idea from earlier,
the purged/embargoed cross-validation splits used elsewhere in this pipeline). The
idea: instead of one fixed-horizon label, draw three barriers around each candidate
entry point and see which one gets hit first.

## Three Barriers, One Race

Picture a price chart with the entry point marked. Above it, a **profit-take**
barrier. Below it, a **stop-loss** barrier. And a **vertical** barrier some fixed
number of bars into the future — a deadline, not a price level. Starting from the
entry, walk forward bar by bar:

- If price clears the profit-take barrier first, label the trade a win (`+1`).
- If price hits the stop-loss barrier first, label it a loss (`-1`).
- If neither happens before the deadline, label it a timeout (`0`) — no verdict, the
  trade just... ran out the clock.

This is a *first-passage* problem, the same general shape as "how long until a random
walk crosses a boundary" that shows up all over physics and queueing theory, and it's
a much more honest description of how a trade actually ends than "what was the price
N bars later, no matter what happened along the way."

## Making the Barriers Tell the Truth About Cost

Having built two whole posts' worth of appreciation for spread and swap fees, our
heroine was not about to let them sit this one out. The barriers aren't placed
against the *raw* price move — they're checked against the **net-of-cost** return.
Concretely, for each candidate entry, at every bar going forward: take the raw
percent move so far, subtract one round-trip spread charge (paid once, at entry),
subtract the swap cost for every 5pm-New-York rollover *actually crossed* since entry
(reusing the very same DST-aware counting logic from the last post), and *that*
adjusted number is what gets compared against the profit-take and stop-loss
thresholds.

The clearest way to see why this matters is a genuinely tiny example that's also a
real test in the code: a price path that rises exactly 1.5% clears a 1.0% profit-take
barrier easily when costs are zero. Charge that same identical price path a 1% round-
trip spread, and the net move is only 0.5% — the barrier is never touched, and the
trade times out instead of winning. Same price, same barrier, opposite verdict,
purely because one version pretended trading was free and the other didn't. A model
trained on cost-blind labels is being taught to recognize moves that look like wins
on paper and often aren't once you actually try to trade them.

## Counting Toll Booths, Carefully, Again

The swap-cost accounting deserved its own scrutiny, since it's genuinely easy to get
subtly wrong: a rollover fee is charged once per calendar night a position is held
through 5pm in New York, in *local* New York time — which means the correct answer
depends on the calendar, not just elapsed hours, and has to survive daylight saving
transitions without a hiccup. There's a dedicated test that walks a position straight
through the exact week the U.S. springs forward, confirming the count comes out right
on both sides of the clock change. It's the kind of test that looks paranoid until
you remember that getting it wrong means either silently overcharging or silently
undercharging every multi-day trade in the entire dataset.

## A Tool, Not (Yet) a Policy

Here's the part our heroine wants to be very clear about, mostly to keep herself
honest: this triple-barrier labeling exists right now as a **standalone research
tool**. It is not wired into the production pipeline. The model currently being
trained still uses the original fixed-horizon target. Swapping the production label
over to triple-barrier outcomes is a real, separate decision — it means choosing new
hyperparameters (how far away should the profit-take and stop-loss barriers be? how
many bars is a reasonable deadline?), retraining, and re-validating against the
existing baselines from scratch. That's future work, deliberately scoped out of this
round, rather than something quietly smuggled in under the same commit as "added a
labeling function."

Next up: recruiting some backup. Our heroine's existing screening tools were built to
ask "does *this* column help predict *this* pair's future" — it turns out the exact
same tools work just as well asking whether a *different* currency pair's columns
help too, and along the way, a few new informants join the investigation.

---

**AI Use Statement**: Claude Code implemented the triple-barrier labeling function
and its DST-aware rollover-counting logic described in this post, and drafted this
post's prose; the decision to keep it a standalone research tool rather than wiring
it into production was made by the author.
