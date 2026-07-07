# Our Heroine Picks a Fight She Can Win

Our heroine's model, having survived exploding gradients, a forward pass that
overflowed to infinity, and a second, sneakier collapse twelve epochs later,
finally finished a clean training run start to finish. Thirteen epochs, early
stopping doing its job, best weights restored. She pulled up the test metrics
expecting, at minimum, a small moment of triumph.

The model beat the majority-class baseline. It lost — badly — to a rule that
amounts to "assume the next few hours look like the last few." A coin that
knows nothing about currency markets, flipped in the shape of yesterday's
pattern, out-forecast a five-layer LSTM that had just survived two separate
near-death experiences to get here.

This is the point in a story where a less careful engineer looks away. Our
heroine, instead, asked the more useful question: was the model bad, or was
the *target* unwinnable?

**The Target Was the Problem**

Everything the diagnostics had already turned up pointed the same direction.
Raw `return` — the signed, bar-to-bar price change this whole system had
ultimately been trying to forecast the sign of, several bars out — had
already been measured, chapters ago, as statistically indistinguishable from
noise at every lag checked, out to twelve and a half days. Not "hard to
predict." *Uncorrelated with itself.* That's not a modeling failure; that's
what a reasonably efficient market is supposed to look like.

Worse, the one thing that had looked like an advantage — the old target's
suspiciously strong lag-1 autocorrelation, and the persistence baseline's
suspiciously strong 63% accuracy riding on it — turned out to be mostly a
trick of arithmetic rather than a market regularity. The target was built as
a return over the *next four bars*, which means two neighboring
training examples share three of those four bars in common by construction.
Of course consecutive labels correlate; they're built from almost the same
data. The persistence baseline wasn't reading the market. It was reading the
plumbing.

**A Target That Actually Has Memory**

Meanwhile, sitting quietly in the same feature set the whole time, was
`volatility_lead` — the realized high-low range over that same forward
window, a measure of how *big* the next few bars' moves would be rather than
which *direction* they'd point. This one had already shown its hand in an
earlier investigation: multi-year regime drift that tracked real market
history, and — the detail that mattered most — autocorrelation that didn't
politely decay and vanish the way a simple, well-behaved process should.
It stayed meaningfully above the noise floor out past a hundred and fifty
lags. Not a coincidence, and not new physics: this is the well-known
asymmetry in financial forecasting. Direction is close to a coin flip.
*How big the moves are* clusters, persists, and has been the entire premise
behind an entire family of models (GARCH and its relatives) for decades.

Switching targets meant asking the ACF/PACF diagnostic the same question all
over again, now pointed at the new target — and the answer flipped
completely. Where the old target's genuinely useful lookback window was a
handful of bars against a configured `n_back` of two hundred, the new
target's practical memory held up almost all the way out to where `n_back`
already sat. For the first time in this whole saga, `n_back=200` looked like
a number somebody could actually defend.

**A Config Change, Not a Rewrite**

Here the earlier modernization work finally paid a very specific dividend.
Every stage of this pipeline reads its target column from one line in
`params.yaml` — nothing about which column gets predicted was ever baked
into source code. Changing the target was, quite literally, changing one
word.

Almost literally, anyway. One tool had quietly assumed the old target's name
as a hardcoded fallback: the autocorrelation diagnostic's default column
would have kept checking the old target even after the config moved on,
silently answering a question nobody was asking anymore. A small, honest fix
— fall back to whatever the config says, not a name frozen in an argument
default — and the single-knob promise was actually true, not just
aspirationally true.

One more thing came along for the ride. With the target reframed around
volatility, the trading-session flags — Tokyo, London, New York, and their
overlap — earned their place in the feature set outright: an earlier
investigation had already shown session timing explaining a real chunk of
volatility's own daily rhythm. Our heroine considered dropping the older
cyclical time-of-day encoding as redundant with the new flags, checked
first rather than assumed, and found the two features weren't nearly as
redundant as they looked — the cyclical encoding still carried real
information the four coarse session buckets couldn't see, including two
entire hours of the day that don't belong to any named session at all. Both
stayed.

**Why This Is a Better Fight, Not a Consolation Prize**

It would be easy to read "we stopped trying to predict direction" as giving
up. It's closer to the opposite: it's aiming the model at the part of the
problem that actually has an answer, and — this part matters just as much —
volatility forecasts are genuinely useful to a trader, just in a different
role than a buy/sell signal. Knowing that the next few hours are about to
get louder lets you size a position down before it gets expensive to be
wrong, tighten a stop before a quiet market stops being quiet, or trigger a
risk circuit-breaker ahead of the fact instead of after the account
statement already shows why it should have fired. None of that requires
knowing which way the market moves. It requires knowing how hard it's about
to move — which, as it turns out, is the one part of this whole system that
was ever genuinely trying to tell us something.

**AI Use Statement:** Claude Code implemented the changes described in this
post (the diagnostics default-column fix, the `params.yaml` target switch,
and the session-flag/cyclical-encoding feature analysis) and wrote this
post's prose itself, across an extended collaborative session with the
author, who directed the investigation, chose the new target, and reviewed
each finding as it came in.
