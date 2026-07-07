# Our Heroine Finds a Ghost at Lag 24

Every previous chapter of this saga had been fought against synthetic data —
tidy, well-behaved rows our heroine had planted herself, precisely so nothing
inconvenient could happen. This week, for the first time, real data hit the
pipes: eleven years of actual EUR/USD hourly candles, pulled straight from
production. The machine groaned (the original notebooks, it turned out, had
asked for 70-100 gigabytes of Spark memory and meant it; the modernized
version had quietly forgotten to ask for any of it, defaulting to Spark's
stock 1 gigabyte and promptly falling over — a whole separate, short, and
slightly embarrassing story). Once that was sorted out, though, our heroine
finally had what she'd been missing this whole time: something real to point
her diagnostics at.

What came back was stranger, and more interesting, than she expected.

**Exhibit One: A Window Twenty Times Too Large**

`n_back=200` — how many hourly bars of history the model looks back across —
had been carried over from the original notebooks with no empirical
justification whatsoever. The ACF/PACF diagnostic said, bluntly: the target
column's own linear memory runs out after about **6 bars**. Not 60. Six.
Worse, when our heroine went looking for a legitimate excuse for the long
window — some hidden daily or weekly rhythm quietly reappearing further out,
the kind of thing a first glance at short-range decay would miss entirely —
she checked the correlation at 24, 48, 72, 120, and 168 bars (one day, two
days, three days, five days, a full week) and found precisely nothing.
Noise. The window really was just too big, with no secret reason hiding
behind it.

**Exhibit Two: A Feature That Won't Sit Still**

`volatility`, meanwhile, failed its stationarity check in the most annoying
possible way: the two tests disagreed. ADF said "stationary enough." KPSS
said "absolutely not." Rather than shrug and pick a side, our heroine went
looking for *why* they disagreed, and found two real things:

First, actual regime drift — yearly average volatility swinging by roughly
2x across the history (a notably calm 2019, a notably turbulent 2022, matching
what anyone who lived through those years in FX markets would recognize
immediately). Re-running KPSS with a trend term instead of a flat level
barely budged the verdict, which ruled out "oh, it's just a simple trend" as
the tidy explanation.

Second, something odder: the shape of volatility's own autocorrelation
doesn't decay the way a simple AR(1) process — the kind of process ADF's own
regression had just estimated, with a tidy `phi_hat=0.897` and a 6.4-bar
half-life — actually should. A process that persistent should be down to
roughly 0.07 correlation by lag 25. Volatility was still sitting at **0.385**.
That's not short memory decaying on schedule. That's the signature of
*long memory* — a well-documented, slightly spooky property of financial
volatility where influence fades far more slowly than any simple exponential
model predicts, discovered independently by people studying markets long
before this project existed.

**Exhibit Three: An Alibi, Then a Confession**

Raw `return`, for its part, had an airtight alibi — dead white noise at every
lag out to 300 (about twelve and a half days), matching the classic
efficient-markets expectation that you can't predict *direction* from past
returns. But `|return|` and `return²` — proxies for volatility rather than
direction — confessed to the exact same strange pattern volatility itself
had shown: autocorrelation dipping toward zero around lag 10, then
mysteriously climbing back up around lag 25 before slowly fading. Finding the
identical fingerprint twice, in two independently-computed series, was the
moment this stopped looking like a fluke and started looking like a lead
worth chasing.

**The Confrontation**

So our heroine zoomed in, and the "resurgence somewhere around lag 25" turned
out to have a precise address: a clean, unmistakable peak at **exactly lag
24** — one trading day, on the nose. That's not a coincidence a
statistician gets to ignore.

A quick table of mean volatility by hour-of-day made the culprit obvious:
volatility sits quiet overnight (roughly 0.0008–0.0012), climbs sharply
around hour 6-7 UTC as London opens, peaks near hour 14 (right in the
London/New York overlap, at roughly **three times** the overnight level),
and fades back down after. Breaking it out by the session flags already
built into this pipeline confirmed it in plain numbers: the London/New York
overlap runs 1.67x the volatility of everywhere else, London alone 1.76x,
and — amusingly — Tokyo actually runs *quieter* than average (0.81x), gently
disproving any assumption that "a market being open" automatically means
"a market being loud."

To make sure this wasn't just a good story, our heroine ran one more test:
shuffle every volatility value *within* its own hour-of-day bucket — same
hour-of-day distribution preserved, but the actual day-to-day sequence
scrambled into confetti — and see how much of the lag-24 correlation
survives. Real data: **0.442**. Shuffled: **0.217**. Just about half of "does
volatility echo itself a day later" turned out to be nothing more mysterious
than "the market has a daily schedule, and tomorrow it'll be open at the same
hours it was open today." The other half is genuine day-to-day memory beyond
the clock — real, just smaller than it first appeared.

**So, Should the Window Shrink?**

Here our heroine resisted the temptation to declare victory and start
deleting code. Every diagnostic above is a *floor*, not a *ceiling* — it
describes what a straight line can see in one column at a time, and an LSTM
fed twenty-five columns simultaneously is allowed to notice things a
univariate correlation coefficient cannot. The lag-24 finding actually makes
the case for a shorter window *stronger*, not weaker, in one specific way:
the model already receives "which trading session is this" directly, at
every single timestep, as its own feature. It doesn't need to stare 24 hours
into the past to work out something it's already being told outright in the
present.

But "the diagnostics suggest a shorter window would work" and "a shorter
window actually works" are two different sentences, and only one of them is
backed by evidence instead of inference. So rather than cut `n_back` from 200
down to whatever the statistics implied and call it a day, the plan is to
actually test it — a modest first step down to a window covering one full
session cycle, evaluated honestly against the current setting using the
rolling cross-validation and multiple-comparisons machinery built a few
chapters back, specifically so a change like this one doesn't get to hide
behind a hunch.

**AI Use Statement:** Claude Code performed the statistical investigation
described in this post (the ACF/PACF and stationarity diagnostics, the
regime-drift and long-memory checks on `volatility`, the volatility-clustering
comparison against raw `return`, and the session/diurnal analysis pinpointing
and explaining the lag-24 effect, including the within-hour shuffle test) and
wrote this post's prose itself, across an extended collaborative session with
the author, who directed each analytical step and will decide what to do with
the `n_back` recommendation next.
