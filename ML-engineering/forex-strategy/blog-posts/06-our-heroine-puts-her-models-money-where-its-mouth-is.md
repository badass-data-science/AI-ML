# Our Heroine Puts Her Model's Money Where Its Mouth Is

This one stands on its own. You don't need to have read anything else in this
series to follow it — all you need is a healthy suspicion of anyone (including
our heroine) who claims a model "works," and a willingness to ask the one
question that claim always dodges: *works how, exactly, and compared to what?*

## Why Bother? Or: Two Ways to Be Wrong About Being Right

Suppose a model looks at two hundred hours of a currency pair's recent
wobbling and predicts, with what looks like real skill, whether the price is
about to go up, down, or nowhere. Suppose it's *right* fifty-five times out of
a hundred — a genuine, hard-won edge over a coin flip. Does it make money?

Not necessarily. Two completely different things can sink it, and neither one
shows up in an accuracy score.

**First: every trade has a toll booth, and the toll booth doesn't care if you
were right.** Buy a currency pair and you buy it at the *ask* price; sell it
and you sell at the *bid* price; the gap between those two — the **spread** —
is money that leaves your account the instant you enter, win or lose. If the
model's fifty-five-cent edge is smaller than the toll, you have built an
elaborate machine for donating money to your broker with extra steps. Hold a
position across a specific moment — 5pm in New York, when the forex trading
day rolls over — and there's a second toll, the **swap** or **rollover** fee,
charged for the privilege of holding a currency pair's implicit interest-rate
trade open overnight. It's directional (you can be charged going long and
*paid* going short the very same pair, or the reverse) and it only applies to
the specific nights actually crossed, not some vague per-day estimate.

**Second — and this is the sneakier one, because it doesn't require the model
to be bad at all — is that "correct" and "correct enough to profit" can
diverge inside the very same trade.** If a model is right very often but its
wins are small and its rare losses are enormous, the toll booth above can
still bankrupt it. Accuracy asks "was the direction right." Money asks "was
the direction right *by enough, after every toll, often enough, for long
enough to trust the pattern* isn't just this one lucky stretch of history."

A backtest is the piece of software whose entire job is to stop dodging that
question. It takes a model's raw predictions and a realistic bill of costs,
and it simulates — honestly, mechanically, without a warm feeling toward its
own model — what would actually have happened, in money, if every one of
those predictions had been traded for real.

## From a Probability to a Position

A trained model here outputs, for every candle, a probability across three
buckets: this move belongs in the bottom third of typical outcomes, the
middle third, or the top third. Turning that into an actual trade decision is
the first job the backtest does, and it's mechanically simple: bottom third
means go short, top third means go long, middle third means stay flat and
touch nothing.

```python
def predicted_classes_to_positions(pred_proba, min_confidence=0.0):
    pred_idx = np.argmax(pred_proba, axis=1)
    confidence = np.max(pred_proba, axis=1)
    positions = np.where(pred_idx == 2, 1, np.where(pred_idx == 0, -1, 0))
    return np.where(confidence >= min_confidence, positions, 0)
```

The one knob worth pausing on is `min_confidence`. Even when the model does
pick a side, it might do so with 34% conviction out of a maximum possible
100% — barely more sure than shrugging. Set a confidence floor, and every
call that doesn't clear it gets forced flat instead of traded. This turns out
to be exactly the right instinct and, in this project's own history, a
disappointing one to actually test: raising the bar sometimes *reduced* the
win rate instead of improving it, which was the first clue that this
particular model wasn't well-calibrated — that its confidence numbers weren't
reliably telling the truth about how often it was right. A backtest's job
isn't just to grade the model; sometimes it's the thing that reveals the
model was never as self-aware as its softmax output implied.

## What a Trade Actually Costs

Every simulated trade pays the same two tolls a real one would.

**Spread**, first, unconditionally, whether the trade wins or loses — a
full round-trip charge, sized proportionally to how large the position is.
(A half-size position pays half the spread cost and earns half the P&L, not
full profit at a discounted toll — an easy place for a backtest to flatter
itself by accident, and specifically tested against here.)

**Swap**, second, but only for nights genuinely crossed. Counting this
correctly is more delicate than "one flat overnight fee" — the boundary is
5pm *America/New_York* clock time, which shifts relative to UTC twice a year
whenever the US changes its clocks, and a sloppy fixed-UTC approximation would
misjudge exactly the trades that matter most: the ones sitting right at the
edge of a rollover. The actual counting is done DST-aware, in real local time:

```python
def count_rollovers_crossed(entry_ts, exit_ts):
    entry_dt = datetime.fromtimestamp(entry_ts, tz=ZoneInfo("America/New_York"))
    exit_dt = datetime.fromtimestamp(exit_ts, tz=ZoneInfo("America/New_York"))
    candidate = entry_dt.replace(hour=17, minute=0, second=0, microsecond=0)
    if candidate <= entry_dt:
        candidate += timedelta(days=1)
    count = 0
    while candidate <= exit_dt:
        count += 1
        candidate += timedelta(days=1)
    return count
```

An intraday trade on hourly data almost always crosses zero of these; a
multi-day hold accumulates one charge per real night held through, not one
per bar. For anyone who'd rather dodge the swap question entirely, there's a
`flatten_before_rollover` mode: instead of holding through 5pm and paying the
fee, the trade is simply skipped that one time — and the backtest reports
exactly how many trades got sacrificed this way, so that trade-off is a
visible number, not a silent one.

## The Two-Races Problem (Or: How This Almost Went Wrong Quietly)

Here's the part of this story with an actual twist in it.

Every row this system ever labels is secretly the result of two independent
horse races run against the same stretch of future price data: one race
asking "if I'd gone long here, would I have hit my profit target, my stop
loss, or just run out the clock?" and a second, completely separate race
asking the identical question for a short position. Whichever race resolves
first — hits its own profit target before the other side does — determines
the official label for that row: a long win, a short win, or (if neither
race's target fires) a flat "no trade" call.

That label is exactly what a model is trained to predict. But a backtest
isn't just grading whether the model called the label correctly — it's
pricing what would *actually* happen if you took the position the model
called, whether or not that matches the label. And here's the trap: for a
long time, this codebase only ever kept the winning race's numbers around.
The losing race's real result — how it would truly have finished, at its own
real price, on its own real timeline — got thrown away the moment the label
was decided.

Think of a two-horse race where only the winner's official finishing time
ever gets written down. If you'd bet on the horse that lost, the record book
doesn't actually know how badly (or well!) your horse did. It just hands you
the winner's time and calls it close enough. Most of the time, for reasons
tied to how symmetric profit targets and stop losses tend to resolve, it *is*
close enough. But not always — and a model is wrong roughly half the time in
any interesting problem, which means this wasn't some rare edge case. It was
the pricing logic for something like half of every trade a model actually
made in every single backtest this project ever ran.

The fix, once spotted, was conceptually simple even if it touched several
layers of code: keep *both* races' real outcomes, always, for every row, not
just whichever one happened to win. A model's wrong-direction bet then gets
priced using the true result of the side it actually took, not a borrowed
number from the side that happened to resolve first.

```python
raw_return_pct = np.where(positions > 0, long_raw_return_pct, short_raw_return_pct)
```

One line, doing the one thing that matters: select each row's true outcome by
what was actually bet, not by what happened to win. Quantifying the damage
*before* fixing it turned out to be its own useful discipline — the worst
possible version of this mistake (a losing bet that would genuinely have won,
had its true result been kept) never actually occurred across many thousands
of real rows checked; the typical mispricing was small, real, and worth
fixing, but not the explanation for why the strategy overall wasn't
profitable. Which is its own kind of good news: the bug was real, but it
wasn't the reason nothing here has made money yet. That honest, unglamorous
distinction — "this was wrong, and here's exactly how much it mattered, and
here's what it *didn't* explain" — is the whole discipline this kind of
software exists to enforce, including on itself.

## Sizing the Bet

Direction is one decision; how much to risk is a separate one, and this
system keeps them genuinely separate. By default every trade is full size.
Optionally, size can instead track realized volatility — scaled down when
the market's been choppy lately, scaled up (up to a ceiling) when it's been
calm:

```python
size = clip(target_volatility / realized_volatility, 0, max_size)
```

`realized_volatility` here is a real, already-observed, backward-looking
average of how big recent candles actually were — not a second model's
prediction. That's a deliberate simplification with its own small history:
an earlier version of this idea used a *second*, separately trained model to
predict volatility ahead of time, purely so its ordinal low/medium/high call
could scale position size. When that second model got retired from the
training pipeline, this fell back to the honest, already-observed number
instead of inventing a substitute prediction — sizing risk off of what
volatility has actually been running, not a guess about what it's about to
be.

Worth calling out, because it's the kind of thing that only becomes obvious
in hindsight: a size must be non-negative to mean what it's supposed to mean.
Every place in this code that picks "which side's true numbers apply to this
row" does so by checking whether the *raw* direction is positive or negative
— not the direction after sizing is applied. Feed it a negative size and,
in principle, you could flip a trade's real economic direction without the
rest of the pricing logic ever finding out, silently reproducing a smaller
cousin of the two-races bug above. Nothing in this project has ever actually
tried to pass a negative size — the one real source of sizing always clips to
zero or above — but "nothing has tried it yet" is a different claim from
"it's impossible," so this is now rejected outright rather than left as a
quiet assumption.

## What Comes Out the Other End

A completed simulation reports, per confidence threshold or position-sizing
scheme tried:

- **how many rows had a trade at all** versus sat flat,
- **win rate** — the fraction of trades that were profitable *after* every
  cost above, not just directionally correct,
- **gross P&L** — the money made or lost before any toll,
- **cost** — the tolls actually paid,
- **net P&L** — the number that actually matters,
- and the full **per-trade net P&L array**, not just the aggregate, because
  an aggregate alone can hide a lot. (A coin-flip win rate with a positive
  aggregate profit, for instance, usually means a few large wins are
  propping up a lot of small losses — a real pattern, but one that needs its
  own dedicated significance test on the per-trade numbers, not just the
  usual win-rate check, before anyone gets excited about it.)

That last parenthetical isn't hypothetical. It's exactly what happened the
first time a promising-looking number showed up here: win rate barely above
fifty-fifty, pooled money positive across every threshold tried, and — once
tested properly — indistinguishable from noise anyway. The backtest doesn't
just report a verdict; it hands you the raw material to interrogate that
verdict yourself, which is the entire point.

## How It's Actually Put Together

The whole simulation is vectorized `numpy` arithmetic over the full test set
at once — no per-row Python loop, since every row's trade is independent of
every other row's. Inputs and outputs are plain arrays and one small
`dataclass` (`BacktestResult`); there's no hidden state, no simulated broker
object, no order book. That simplicity is deliberate: a P&L simulator that's
complicated enough to need its own careful debugging has quietly become a
second research project competing with the one it's supposed to be
evaluating.

The backtest doesn't stand alone, though — it's the third stop for data that
started somewhere else entirely. A sibling project (`forex-ML`) pulls raw
candlestick data, engineers features, and trains the actual prediction
model; as part of that, it runs the two-races labeling scheme described
above and holds out a genuine test set the model never touched during
training. A trained model's predictions on that held-out set, along with
each row's true price, spread, and — critically — each side's own real
outcome, get logged as an artifact to `MLflow` (an open-source experiment
tracker) alongside the model itself. This project's `run_backtest.py` is the
bridge: it looks up the right registered model version for a given
currency pair and time granularity (filtering out, along the way, an
entirely different kind of model that used to share the same registry
namespace — a bug this project's model-lookup code was built specifically
to close), downloads its logged predictions, and hands them to the
simulation described above. Keeping this as three separate projects rather
than one big one means the modeling side never has to think about spreads
and swap fees, and the trading side never has to know how an LSTM gets
trained — each piece can be reasoned about, and tested, on its own terms.

## References

- Marcos López de Prado, *Advances in Financial Machine Learning* (Wiley,
  2018) — the triple-barrier labeling method underlying the "two races"
  design above, and the broader argument that a label should describe how a
  real trade actually resolves (target hit, stop hit, or timeout) rather
  than a fixed-horizon price change.
- OANDA's own documentation on financing/rollover charges — the practical,
  real-world source for why swap fees are asymmetric by direction and
  anchored to a specific New York clock time rather than a flat daily rate.
- The standard "inverse volatility targeting" idea in systematic trading —
  size positions down when recent realized risk is elevated, up (to a cap)
  when it's subdued — used here in its simplest, single-instrument form.
- Any introductory treatment of backtesting pitfalls (look-ahead bias,
  survivorship bias, multiple-comparisons/"testing until something looks
  significant") is worth reading before trusting a backtest's output at
  all, including this one's — several of the design choices described here
  exist specifically to avoid a particular flavor of one of these traps.

## Where This Goes Next

A few honest gaps, in roughly the order they're worth closing:

- **Exit-bar spread.** The cost model currently charges the *entry* bar's
  spread for both legs of a round trip, because the exit bar's spread isn't
  currently threaded through from the data-engineering side. Spreads do
  drift with time of day and liquidity, so this is a real, if probably
  small, source of imprecision — worth plumbing through properly rather
  than assuming it nets out.
- **A permanent home for the per-trade significance check.** The right test
  for "is this pattern real" turned out to depend on the *shape* of the
  result (win-rate-driven versus payoff-asymmetry-driven), which took a
  human noticing something looked different to catch. That kind of check
  belongs as a standard, always-computed part of a backtest's output, not
  something reached for ad hoc after a suspicious-looking number appears.
- **Slippage and partial fills.** This simulator assumes every trade fills
  completely, instantly, at the quoted price plus spread. Real execution
  is noisier than that, particularly around news events or thin liquidity.
- **Portfolio-level backtesting.** Everything here evaluates one currency
  pair in isolation. Real capital allocation questions — correlation
  between pairs, shared risk budgets — aren't modeled at all yet.
- **Continued skepticism of the backtest itself.** The two-races bug was
  found by someone asking "but is this actually implemented correctly?"
  after the software had already been trusted for a while. That's not a
  one-time audit that's now finished — it's the right permanent posture
  toward any code whose entire job is telling you whether to believe
  something.

---

**AI Use Statement**: Claude Code designed and implemented every piece of
software described in this post — the position/cost/swap simulation, the
volatility-based position sizing, the model-registry lookup, and the
two-races pricing fix, including finding and quantifying that bug before
touching any code to fix it — and drafted this post's prose from that work.
The author directed the investigation, asked for the backtest to be
independently double-checked (twice) before trusting its numbers further,
and reviews and edits all published material, but did not write the code or
run the analysis herself.
