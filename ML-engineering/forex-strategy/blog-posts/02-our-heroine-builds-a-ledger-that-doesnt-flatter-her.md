# Our Heroine Builds a Ledger That Doesn't Flatter Her

Having identified, in the previous installment, that a model can be "right" and
still lose money, our heroine set about building the thing that would actually tell
her the truth: a **backtest** — code that takes a trained model's predictions and a
realistic set of costs, and simulates what would have happened, in money, if she'd
actually traded on them. She was aware, going in, of a specific occupational hazard
of building your own P&L simulator: the temptation to *accidentally* make it flatter
you. Every design choice below was made with that temptation firmly in mind.

## Finding the Right Model in a Crowded Filing Cabinet

Before simulating anything, she needed to actually load the right trained model.
This turned out to be less obvious than it sounds, because of how the model registry
was organized: every single trained model — every currency pair, every granularity,
every architecture experiment — gets registered under one shared name
(`forex-lstm`), the same way every document in an office might get filed in one
cabinet drawer regardless of what it's about. Each registered *version* in that
drawer is tagged with metadata (which pair, which granularity, a hash summarizing the
model's configuration) so you can dig out the one you want.

Except — and this is the bug our heroine found while building this very post's
subject matter — none of those tags recorded *which target the model was trained to
predict.* Two completely different kinds of model can share a pair: one trained to
predict price direction (`pd_lead`), another trained to predict volatility magnitude
(`volatility_lead`). Without a tag distinguishing them, "give me the latest model for
EUR/USD" could silently hand back whichever one happened to be trained most
recently — direction or magnitude, no way to tell from the outside. She added a
`column_y` tag to close the gap, and then, distrusting her own fix on principle,
wrote a test that deliberately registers the *wrong* target's model with a *higher*
version number than the *right* one, and confirms the lookup still returns the right
one. It did. Filing cabinet secured.

## Turning a Prediction Into a Position

The model's raw output is a probability over three classes — the target value fell in
the bottom third of its historical range, the middle third, or the top third. For a
direction-predicting model, that maps naturally onto a trading decision: bottom third
means "go short," top third means "go long," middle third means "sit this one out."
An optional confidence threshold lets her force a position flat if the model's
top-choice probability is too close to a coin flip — a low-confidence call is exactly
the kind of trade least likely to survive contact with the tolls from last post.

## Charging Herself Rent

This is where the honesty work actually happens. For every row that results in a
trade, the simulator charges the **full round-trip spread** as a percentage of price,
unconditionally — it doesn't matter whether the trade wins or loses, the toll is the
toll. And it charges it in a way that scales consistently: if a position is sized at
half strength (more on sizing in a moment), it earns half the profit *and* pays half
the cost, not full profit with a "convenient" cost discount. That consistency check
mattered enough that there's a dedicated test for it — it's exactly the kind of place
a backtest can quietly lie to itself by accident.

## The Toll Booth at 5pm

Then there's the overnight toll. The simulator can charge a configurable
swap-cost-per-night, but — and this took real care — only for nights *actually
crossed* between a trade's entry and exit, not a flat "per bar held" fee. An intraday
trade on hourly data usually crosses zero 5pm-New-York boundaries; a multi-day trade
crosses one per night. Counting this correctly meant reusing a small, carefully
tested piece of forex-ML's own labeling code (`count_rollovers_crossed`) that counts
5pm boundaries in *real* America/New_York time — daylight saving and all — rather
than a fixed UTC offset. An hour's sloppiness here is the difference between a trade
being charged a night's swap or not; it's not a "close enough" kind of approximation.

For traders who'd rather not deal with the overnight toll at all, there's an
alternative: a **flatten-before-rollover** mode. Instead of holding through a 5pm
crossing and paying swap, any trade that *would* cross one gets skipped
entirely — the backtest reports exactly how many trades were sacrificed this way, so
that choice is visible, not silent.

## Letting Volatility Hold the Steering Wheel

Finally, position size doesn't have to be all-or-nothing. If a *second*, separately
trained model exists for the same pair — one trained to predict volatility rather
than direction — its predicted volatility tercile can scale the trade size down as
predicted risk rises: full size when volatility is predicted low, progressively
smaller as it's predicted medium or high. This is the standard "size inversely with
risk" idea from volatility targeting, applied honestly to what the model can
actually tell you (an ordinal low/medium/high call) rather than inventing a precise
number it was never trained to produce. Before combining the two models' outputs,
the code checks that their test sets are lined up row-for-row by timestamp — a subtle
place two independently trained models could otherwise get silently mismatched.

None of this makes for thrilling reading, which is rather the point: a P&L ledger
that's interesting to read is usually one that's cutting corners somewhere. The next
post covers a genuinely new idea rather than careful bookkeeping — teaching the
system a better definition of what "winning a trade" even means.

---

**AI Use Statement**: Claude Code designed and implemented the backtest engine
described in this post, including the position-sizing consistency check and the
model-registry bug fix, with the author reviewing and directing each design decision;
Claude Code also drafted this post's prose.
