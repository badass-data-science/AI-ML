# Our Heroine Retires the Buzzer

Over on the trading side of this story — a chapter this blog hasn't gotten
to yet in this feed, but one our heroine kept thinking about anyway — she'd
built a proper cost-aware labeling scheme called the triple-barrier method:
watch a trade forward from its entry, and call it a win, a loss, or a
timeout based on whichever of three barriers gets hit first, net of real
spread and swap cost. She'd built it carefully, tested it thoroughly, and
then, quite deliberately, left it sitting on a shelf. The model actually
being trained still used the old scheme: whatever the price happened to be
doing exactly four bars from now, buzzer or no buzzer.

That was a defensible thing to defer at the time. It stopped being
defensible the moment she asked herself the obvious follow-up question out
loud: *if a real trade doesn't wait around for a fixed buzzer to sound, why
is the training target still the one thing in this whole pipeline built
like it does?*

**Wait, Didn't We Already Lose This Fight?**

Here's the part worth being honest about, because skipping it would make the
rest of this chapter sound like it forgot its own history. A few chapters
back, this same target — "the percent price change exactly four bars from
now" — got benched for a specific, well-earned reason: raw `return` turned
out to be statistically indistinguishable from noise at every lag checked,
and the persistence baseline's suspiciously strong showing on that target
turned out to be mostly mechanical — two neighboring four-bar windows share
three of their four bars by construction, so of course they correlate.
That wasn't the market. That was the plumbing. Direction, at that fixed
four-bar horizon, was a fight nobody was winning, so the target got switched
to something with real memory instead: forecasting *how big* the next
several bars' moves would be, not which way they'd point.

So doesn't reintroducing a directional, three-class label — win, loss, or
timeout — just walk straight back into the fight that was already lost?

It doesn't, and the reason why is worth sitting with, because it's not just
a rhetorical dodge. "Will the raw price be up or down in exactly four bars,
no matter what happens in between" and "will a real trade, with a real
profit target and a real stop-loss, both net of real trading cost, hit its
target before it hits its stop" are two different questions wearing similar
clothes. The first is a brittle, symmetric bet on the sign of a number
already shown to behave like noise at that exact horizon. The second is
coarser and asymmetric on purpose, comes with a built-in "no clear answer
yet" option that the old scheme never had, and — critically — it isn't
locked to a fixed distance in time at all. A trade in this new scheme can
resolve in one bar or in twenty-four; the old target called it after
exactly four, whatever "it" was doing. Whether this new bet turns out to be
winnable is a question for an actual training run, and deliberately not one
this chapter answers — no production model has been retrained on the new
target yet. What this chapter *can* say honestly is that it isn't the same
bet as the one that already lost.

**One Target, Not Two**

The old scheme actually had two selectable targets sharing one config knob:
the directional one just discussed, and the volatility one that replaced it
in production. The question of what triple-barrier should replace turned
out to need answering directly rather than assumed, and the answer was
*both* — not layering triple-barrier in as a third option alongside the
other two, and not just swapping out the directional one while leaving
volatility-forecasting alone. That meant the volatility target's actual
job — feeding position size, not being a headline prediction in its own
right — needed a real replacement, not just a shrug. More on that shortly;
it turned into the nicest idea in this whole chapter.

**A Label That Already Knew Its Own Answer**

The old scheme needed a genuinely fiddly step: take a continuous percent
change, and cut it into terciles — lowest third, middle third, top third —
using thresholds fit on the training set alone, so the model saw a discrete
short/flat/long class instead of a raw number. Triple-barrier's label
doesn't need any of that. A trade either hits its profit-take (call it
`+1`), hits its stop-loss (`-1`), or times out (`0`) — it arrives already
discrete, no percentile-fitting machinery required. Mapping it onto classes
turned into the easiest design decision in the whole migration: `-1` becomes
class 0, `0` becomes class 1, `+1` becomes class 2 — which happens to be
*exactly* the short/flat/long convention the trading side's backtest was
already built around. Nothing downstream had to change to accommodate it.
The label and the strategy code were already speaking the same language;
they just hadn't been introduced.

**Two Numbers That Look Alike and Aren't**

One subtlety needed real care. The triple-barrier code already tracked a
`net_return_pct` — the realized move at the exit bar, with spread and swap
already subtracted out. It's tempting to hand that straight to a backtest
as "the outcome." It's also wrong: the trading side's own backtest charges
its *own* spread and swap on top of whatever it's handed, which means
feeding it a number that's already net of cost double-charges every single
trade. The fix was a second, parallel field — `raw_return_pct`, the same
exit-bar move *before* any cost gets subtracted — and that's the one that
actually flows downstream now. A quiet reminder that "net of cost" and "the
number a backtest should use" are not automatically the same number, even
when they're one line apart in the same function.

**Replacing a Model With a Memory**

This is the part worth calling the actual payoff of the whole chapter.
Position sizing used to lean on a *second*, separately trained model —
predicting an ordinal low/medium/high volatility class for the next window,
purely so a trade could be sized down when things looked likely to get
loud. That always felt like more machinery than the question deserved, and
an earlier investigation had already quietly proven why: volatility, unlike
direction, has real memory. A shuffle test back then found that about half
of "does volatility echo itself a day later" survives even after scrambling
away the daily clock cycle — genuine, non-mechanical persistence, not just
plumbing dressed up as a pattern.

If recent volatility genuinely predicts near-future volatility, there's no
principled reason to train a whole second network to forecast it. You can
just look behind you. The fix was a new column — a plain twelve-bar trailing
average of realized bar-to-bar range, computed once in feature engineering
and carried through untouched as reference data, never fed to the model as
an input. No second model to train, no second model to register, no second
model's test set to keep timestamp-aligned with the first. The volatility
target didn't get replaced with a better prediction. It got replaced with
the observation that, for volatility specifically, the recent past already
*is* a pretty good prediction — which is exactly what that old shuffle test
was trying to tell everyone months ago.

**A Bug Caught Before It Was Born**

The first instinct was to reuse a column already sitting in feature
engineering — a twelve-bar moving average of volatility, computed whenever
the configured moving-average windows happen to include twelve. Reusing it
felt free. It wasn't: every single test in this project's suite configures
those windows as `[3, 5]`; only the real production config reaches for
`[12, 30, 50]`. Hardcoding a reference to "the twelve-bar column" would have
quietly assumed a column that doesn't exist under most of this codebase's
own test fixtures — the kind of bug that hides successfully until the one
day someone runs the real pipeline against real data and watches a Spark
job fail on a column name nobody remembers depending on. Building a small,
unconditional, always-computed column instead — independent of whatever
windows happen to be configured for anything else — cost one extra line of
code and avoided a bug that would have taken considerably longer to explain
after the fact.

**What's Still Left Marked "Placeholder"**

None of this chapter pretends to be a finished, tuned system. The
profit-take and stop-loss thresholds, the maximum holding period, and the
swap-cost-per-night figure are all first guesses, explicitly flagged as such
in the config itself — not yet checked against this pair's actual
distribution of moves the way `n_back` eventually was against real ACF/PACF
diagnostics. The swap cost, specifically, is a configured constant rather
than a live rate pulled from the exchange, on purpose — real rate ingestion
already exists on the trading side of this story, and wiring it in here is
a deliberately separate next step, not something worth smuggling into a
chapter about relabeling. And no production model has actually been
retrained against any of this yet. This chapter changed what the target
*is*. Whether it's a fight worth winning is still an open question, and an
honest one is better than a rushed answer to it.

**AI Use Statement:** Claude Code designed and implemented the triple-barrier
migration described in this post (the label-to-class mapping, the
`raw_return_pct`/`net_return_pct` split, the `realized_volatility` feature
and the test-configuration bug it avoided, and the config/diagnostics
updates), and wrote this post's prose itself, across an extended
collaborative session with the author, who asked the question that started
the chapter, decided both targets should be replaced rather than one, and
reviewed each design decision as it came in.
