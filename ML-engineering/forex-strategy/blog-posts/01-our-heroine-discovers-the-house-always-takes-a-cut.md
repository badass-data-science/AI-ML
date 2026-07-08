# Our Heroine Discovers the House Always Takes a Cut

Our heroine had, by this point, spent several very satisfying sessions teaching a
neural network to stare at a wiggly line and guess which way it would wiggle next.
She had switched targets mid-project (more on that in an earlier chapter), fixed an
exploding-gradient problem by clipping things in two different places, and built
herself a small arsenal of statistical tools for deciding which columns of data were
worth the GPU's time. Numbers were going up. Loss curves were going down. Baselines
were being beaten, modestly but honestly. She allowed herself a small, entirely
justified victory lap.

Then she asked herself a rude question: *if this model is right, does anyone actually
make money?*

It is a testament to how absorbing the modeling work had been that this question
arrived so late. "Predict whether the price goes up or down" and "identify a trade
worth taking" sound like the same problem. They are not. A model can be *correct*
about direction more often than not and still lose money on every single trade, for
the extremely boring reason that every trade has to pay a toll before it's allowed to
be profitable. Our heroine had built a very nice weather forecaster and never once
asked whether it was going to rain money or just rain.

## The House's Cut, Part One: The Spread

The first toll is the **spread** — the gap between the price you can buy at (the
"ask") and the price you can sell at (the "bid"). It exists at every broker, for
every instrument, at every moment, and it is not a rounding error. If EUR/USD's mid
price is 1.10000 and the spread is a couple of pips, buying and then immediately
selling back — with *no* price movement at all — loses you money. Every trade pays
this once, round-trip, whether it wins or loses. A model that's right 55% of the
time on moves *smaller* than the spread is not a trading strategy; it's a very
elaborate way to donate money to your broker.

This is the part of "does the model make money" that a classification-accuracy
number simply cannot see. Accuracy asks "was the direction right." Money asks "was
the direction right *by more than it cost to find out.*"

## The House's Cut, Part Two: The Overnight Toll

The second toll is sneakier, because it only applies if you're still holding a
position at a specific moment: **5pm New York time**, when the forex market's
trading day officially rolls over to the next one. If you're holding a position
across that boundary, you're charged (or, occasionally, credited) a **rollover** or
**swap** fee, once per night held.

Where does this fee come from? A currency pair is, financially speaking, a trade of
one country's interest rate for another's — when you go long EUR/USD, you're
notionally borrowing dollars to buy euros, and the swap fee approximates the
difference between the interest you'd earn on the euros and the interest you'd owe
on the dollars. This is why the fee isn't symmetric: going long a pair can cost you
money overnight while going short the *same* pair earns you money overnight, or vice
versa, depending on which currency's rates are higher. It's also why the fee is
charged at a specific clock time in a specific city's timezone (5pm America/New_York,
daylight-saving-time and all), not some vague "once a day" notion — because that's
literally when the trading day is defined to end.

A model — or a human — that ignores this fee will systematically overestimate the
profitability of any strategy that tends to hold positions overnight, and won't even
be wrong in an obvious way. It'll just be quietly, consistently too optimistic.

## Redrawing the Org Chart

Once our heroine had named these two tolls out loud, it became obvious that "does
this make money" was a genuinely different job from "what will this time series do
next" — different enough that it deserved its own home rather than getting bolted
onto the existing model-training code as an afterthought. So the work got split
three ways, matching a division of labor real quant shops already use:

- **`forex-etl`** — data ingestion. Its job is to go get raw information (candlestick
  prices, and now also swap rates, economic calendar events, and retail positioning
  data) and land it somewhere durable. It doesn't know or care what any of it means.
- **`forex-ML`** — research and modeling. Feature engineering, labeling, training,
  and diagnostics. Its job is to turn ingested data into a trained model and tell you
  how much to trust it.
- **`forex-strategy`** — the new project this whole series is about. Its job is to
  take a trained model's predictions and a live set of costs, and decide whether
  there's an actual trade in there worth taking, net of the tolls above. It doesn't
  need Spark, doesn't need to know how to train an LSTM, and — unlike the other
  two — is allowed to think in dollars and cents instead of accuracy percentages.

Keeping these separate wasn't just tidiness. It meant forex-ML could stay focused on
"is this a good model" without slowly turning into a P&L simulator, and it meant the
P&L simulator could be tested and reasoned about on its own terms, using whatever
model happens to be the current champion, without caring how that model was trained.

## The Plan

Over the next several posts, this series covers: building a cost-aware backtest
engine that actually charges itself spread and swap fees (and can choose to just
avoid the swap fee entirely by flattening a position before 5pm); teaching the
labeling process itself to think in terms of "did this trade hit its target, its
stop, or just time out" instead of a fixed, arbitrary lookahead window; and finally,
recruiting some new sources of information — other currency pairs, scheduled economic
announcements, and even what other traders are doing — while being very careful not
to make up numbers that can't be verified.

Spoiler: our heroine does eventually get a working, honestly-costed system. She also
discovers a real bug hiding in her own bookkeeping along the way, because of course
she does.

---

**AI Use Statement**: Claude Code did the investigative and engineering work described
in this post (architecture decisions were discussed with and approved by the author
at each step) and drafted this post's prose from that work.
