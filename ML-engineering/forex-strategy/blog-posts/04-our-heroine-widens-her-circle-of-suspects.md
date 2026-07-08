# Our Heroine Widens Her Circle of Suspects

Several posts back — in a chapter of this saga not covered here but referenced
often — our heroine built herself a small detective agency's worth of statistical
tools for one specific question: *does this column of data actually help predict the
target, or is it dead weight?* Cross-correlation for a cheap first look, Granger
causality for "does this column's history carry information the target's own history
doesn't," a full multivariate VAR model with forecast-error variance decomposition
for when candidates might be stepping on each other's toes, and a Lasso-regularized
regression that zeroes out the useless ones automatically. Four techniques, cheapest
and least rigorous first, all correcting properly for the fact that testing many
candidates at once is itself a statistical hazard if you don't account for it.

All four had, until now, only ever interrogated candidates from *within* one currency
pair's own data. Which raised an obvious question our heroine had somehow not asked
out loud before: *does GBP/USD know something about EUR/USD?*

## Interrogating the Neighbors

It turns out the honest answer to "how do I extend these four techniques to another
currency pair" is: barely any new code. Every major pair already flows through
exactly the same feature-engineering pipeline, each keeping its own address (keyed by
instrument and granularity) in the same filing system used everywhere else in this
project. All that was missing was a function to pull one pair's target column and
*another* pair's candidate columns, line them up by timestamp, and hand the result to
the exact same four techniques, completely unchanged.

The one real wrinkle: every pair's data uses the *same* column names — every
instrument has a `return`, a `volatility`, a `diff_spread_close` — so a candidate
borrowed from GBP/USD gets renamed to something like `GBP_USD__return` before joining,
to keep it from silently colliding with EUR/USD's own `return` column. And because
different pairs can have slightly different histories of gaps and forward-fills, the
join keeps only timestamps present in *every* pair involved, rather than assuming
perfect alignment. From the command line, asking "does GBP/USD's return and spread
movement help predict EUR/USD's volatility" now looks like:

```
--cross-pair-candidates "GBP/USD:return,diff_spread_close;USD/JPY:volatility"
```

— one semicolon-separated group per neighboring pair being interrogated, columns
within a group comma-separated. The screening report that comes back looks exactly
like the single-pair version, because underneath, it *is* the single-pair version;
it just doesn't know or care where its candidates came from.

## Recruiting New Informants

While the existing detectives were learning to interview the neighbors, a separate
effort went into recruiting entirely new sources of information — three of them,
landing in the data-ingestion side of the project rather than the modeling side.

**Swap rates.** The rollover fee introduced two posts ago has to come from
*somewhere* real, not a guessed constant — OANDA publishes per-instrument long and
short financing rates directly, reachable with the exact same login already being
used to pull price candles, just a different web address. The only genuine wrinkle:
finding *which* account to ask required a small fallback (use a configured account ID
if one's on file, otherwise ask OANDA which account the login belongs to and use
that) rather than assuming the answer.

**Economic calendar events.** Scheduled announcements — interest rate decisions,
employment reports, inflation prints — are exactly the kind of "known time, unknown
outcome" event that reliably moves markets, and they are *not* something OANDA
publishes at all. This meant recruiting a genuinely separate informant (a service
called Finnhub) with its own separate credential, kept in its own separate lockbox
rather than mixed in with the OANDA one. A nice detail: a future scheduled event
naturally doesn't have an "actual" outcome yet, only a market estimate — the code
simply leaves that field out entirely rather than pretending it knows a number it
doesn't have yet.

**Retail positioning.** OANDA also publishes an aggregated, anonymized snapshot of
where its own retail traders currently have orders and open positions — a rough
gauge of "crowd sentiment" that's a popular (if controversial) input in a lot of
retail trading folklore.

## The Informant Who Wouldn't Make Things Up

This last one is where our heroine ran into a genuinely interesting temptation, and
it's worth telling honestly because resisting it was the right call. OANDA's
positioning data doesn't arrive as a tidy "68% of traders are long" headline number —
it arrives as a *histogram*, dozens or hundreds of price buckets, each with a
percentage of long positions and a percentage of short positions sitting at that
price level. Turning that into one clean "overall sentiment" number *feels* like the
obviously useful thing to do. It would also have required silently assuming a
specific mathematical fact about how OANDA normalizes those per-bucket percentages —
whether they sum to 100% within each side, or overall, or something else entirely —
that couldn't actually be confirmed without a live response to check against.

Rather than guess and ship a plausible-looking number that might quietly be wrong,
the decision was to store every raw bucket exactly as received and leave the
aggregation to whoever eventually needs it, once that assumption can be verified for
real. It's a less satisfying stopping point than "and then we computed the crowd
sentiment score" — but a confidently wrong number is worse than an honestly
unfinished one, and this project has already spent three posts arguing that costs and
correctness matter more than a good-looking result.

## Where Things Stand

That closes out the initial build: a cost-aware backtest that charges itself real
spread and swap fees, a better definition of what a winning trade even means, a
screening toolkit that now looks sideways at other currency pairs instead of just
inward, and three new sources of market information sitting in the pipeline, ready
for whoever picks up the next chapter. Nothing downstream reads the economic calendar
or positioning data yet — they're ingested and waiting, not yet wired into any
decision. Which is, all things considered, a perfectly good place to pause a
detective story: informants recruited, alibis checked, and one number our heroine
pointedly declined to make up.

---

**AI Use Statement**: Claude Code implemented the cross-pair screening extension and
the three new data-ingestion pipelines described in this post (including the decision
to surface, rather than resolve, the positioning-data normalization uncertainty,
which was then confirmed with the author), and drafted this post's prose.
