# Our Heroine Aces the Test and Bombs the Interview

It started as a throwaway question. Buried in `params.yaml`, one line had been
sitting untouched since the very first draft of this project: the dense layer
after the LSTM stack had exactly 7 units in it. Not 8. Not a round number like
16 or 32. Seven. Our heroine was asked where that number came from, and had to
admit: she didn't know. It was inherited verbatim from the original
hand-written script, with no comment, no reasoning, nothing. Just a 7,
sitting there since day one, never once questioned.

"Let's test it properly," she said, the way people say things right before a
very long week.

**The Sweep That Went Nowhere**

The plan was simple, and reused a trick from a few chapters back: hold
everything else fixed, vary only the dense layer's width, and run the same
rolling-CV comparison that had already found a real, non-obvious sweet spot for
how *deep* the network should be. Surely width would tell a similar story.

It didn't. Seven, sixteen, thirty-two, sixty-four, a hundred and twenty-eight —
every single one landed in the same narrow, noisy band, and none of them beat
the simplest possible baseline: always guess whichever class shows up most
often in the training data. The inherited 7 was exactly as good, or exactly as
bad, as anything four times its size. Widening a layer that isn't the
bottleneck doesn't do anything. Lesson filed. Or so it seemed.

**The Suspiciously Good Baseline**

Buried in that same comparison was a second number that should have been
boring and instead was deeply strange. Alongside "always guess the majority
class," there's a second trivial baseline: "guess that this period repeats
whatever the previous period actually turned out to be." For most of this
project's history that number had been a mild, sensible baseline. In this
comparison it scored 85%. The model, trained on real features with real
effort, scored 35%.

That's not a baseline anymore. That's a baseline wearing a disguise.

Digging in turned up a real bug, not just "the market is more autocorrelated
than we thought." That baseline was predicting each row using the *actual,
already-known outcome* of the row before it — except under this project's
labeling scheme, a label isn't actually resolved the instant it's created. It
takes a while to find out whether a trade would have won, lost, or gone
nowhere — a mean of about fifteen hours, out of a maximum allowed window of
twenty-four. Which meant that "the previous row's actual outcome," used to
predict a row barely an hour later, usually hadn't happened yet from that
row's point of view. The baseline was quietly borrowing information from up
to twenty-three hours in the future and calling it a prediction.

Fixed properly — only ever persist from a label that has genuinely, causally
already resolved by the time you'd need it — that suspicious 85% collapsed to
38%. Barely above the boring baseline it was supposed to be a step up from.
Which meant the real comparison, this whole time, hadn't been "can the model
beat a strong baseline." It had been "can the model beat a baseline that was
cheating." A very different question, and a much more honest one.

**A Gut Check Before Any More Dial-Turning**

With that fixed, the actual state of things looked worse than expected: no
width had ever beaten even the honest majority-class baseline. Before turning
any more dials — dropout, learning rate, all the usual suspects — it seemed
worth asking a more basic question first: was this a model that genuinely
couldn't learn anything, or a model that learned the training data just fine
and simply didn't generalize?

The answer was sitting right there in already-collected training logs, no new
runs required. Training accuracy climbed steadily, epoch over epoch, well
past chance — real learning, no question. Validation accuracy did not follow.
It sat flat and noisy the entire time, the gap between the two widening
steadily. That's not a broken model. That's a model with real capacity,
overfitting on far too little data — these comparisons were only using a
sliver of the real history to keep each experiment fast.

**Does More Data Help? (Two Steps Forward, One NaN Back)**

The obvious next question: would the gap actually close with a proper amount
of data behind it? A single larger test, roughly double the size, ran
straight into this project's oldest, most personal nemesis — the training run
diverged to NaN partway through, saved only by the batch-checkpoint recovery
mechanism built specifically for this failure mode a while back. The
recovered snapshot happened to beat both baselines nicely, but it had only
survived three real epochs before catching fire, so nothing could be
concluded from it cleanly.

Running it again with a second, genuinely different slice of history
answered the more important question: the exact same window diverged to NaN
again, deterministically. Not a fluke — a real, reproducible fragility tied to
something specific in that stretch of the market's history. A fresh window
that didn't share that problem trained cleanly the whole way through, and
still edged out the majority baseline, just less dramatically. More data
genuinely helped. It just wasn't a clean, unconditional win.

While in there, a second question got tested alongside it: would bigger
training batches smooth over the instability? They did — no divergence at
all with four times the batch size. But the model that emerged from that
stability was a different kind of failure: it had simply learned to guess the
majority class, every single time, on both windows tested. Trading one honest
risk (occasional numerical blow-ups, already survivable) for a different,
worse one (a model that quietly gives up and mimics the boring baseline) is
not a trade worth making.

**Time for the Real Thing**

Enough diagnostics. Architecture wasn't the lever, batch size wasn't the
lever, and the data-size experiments pointed the same direction the whole
time: run the real thing, on the real amount of history, and see what
happens.

The first real attempt diverged to NaN — expected, the recovery mechanism
handled it exactly as designed — and then crashed anyway, for a completely
unrelated reason, right at the very last step of logging the model. That
turned out to be a genuine bug in a piece of third-party plumbing this
project depends on: an internal bookkeeping step tries to copy a metric that
was already recorded, and if that metric happens to be a NaN (exactly the
kind you'd have on hand after a divergence), the library's own safety check
for "did I already save this" quietly breaks, because NaN famously refuses to
equal itself even when compared against its own twin. The fix took one line —
tell that bookkeeping step to look somewhere nothing was ever recorded, so it
has nothing to trip over — confirmed by deliberately reproducing the exact
crash first, then watching the fix make it vanish.

Run two, on the real ~50,000-row history: no divergence, no crash, a model
trained start to finish. Final scoreboard: the model at 43%, the honest
majority baseline at 33%, the fixed persistence baseline at 38%. A real,
clean win over both, checked with a proper paired significance test rather
than eyeballed — the kind of result so unlikely to be chance that the
probability of it happening by luck alone has around 49 zeros after the
decimal point.

**The Diploma Doesn't Get You the Job**

This felt like the finish line. It was not the finish line.

Being right about the *class* more often than a coin flip says nothing, on
its own, about whether following the model's calls would make money — those
are genuinely different questions, and it was time to actually ask the second
one. Every signal the model produced got run through a real, cost-aware
backtest: real spread, real live swap rates, no hand-waving.

The verdict: on the trades it actually recommended taking, the win rate was
49.2%. A coin flip, with extra steps. And once real trading costs were
totaled up across every one of those trades, the net result was a loss large
enough to make the point without any ambiguity at all.

A model can be a genuinely better *classifier* than a trivial baseline —
proven, significant, no asterisks — and still not be a trader anyone should
follow. Those are different exams, and this model aced one and bombed the
other.

**Turning Up the Confidence Dial (Which Made Things Worse)**

There was one obvious lifeline left to try: maybe the model has a real edge
buried inside its most confident calls, drowned out by a mass of low-conviction
noise trades. Filter down to only the predictions it feels strongly about,
and the win rate should climb.

It didn't. It got worse, in a straight line, the more the confidence bar was
raised — and past a certain point, there weren't any trades left to even
check, because the model never once felt more than 50% sure about anything,
on any row, the entire test set. A well-behaved model should get more right,
not less, the more selective you are about which of its opinions you trust.
This one does the opposite. Whatever "confidence" its output layer reports
isn't tracking whether it's actually correct — which means there's no hidden
gem to go filter for. The model isn't a good trader with some noise mixed in.
It's a mediocre classifier with no reliable way to tell its good guesses from
its bad ones.

**Where Things Stand**

A lot got fixed this chapter, and all of it needed fixing regardless of how
the ending turned out: a leaking baseline that made everything upstream of it
look worse than it was, a real numerical-stability question about batch size
answered rather than guessed at, and a genuine third-party bug caught,
reproduced on purpose, and closed for good. The pipeline is more honest than
it was at the start of this chapter, on every axis that matters.

What it does not have, yet, is a model worth following into a trade. Ending
one chapter with "the numbers look great" would have been the tidier
story. Ending it with "the numbers look great, and here's the cost-aware
backtest proving that doesn't matter yet" is the more useful one — and the
one that's actually true.

**AI Use Statement:** Claude Code ran this entire investigation described in
this post — the dense-width comparison, diagnosing and fixing the
persistence-baseline information leak, the train/validation overfitting
diagnostic, the data-size and batch-size follow-up experiments, diagnosing
and fixing the MLflow logging crash, running the full training run, the
statistical significance test, the cost-aware backtest, and the
confidence-filtering follow-up — and wrote this post's prose. The author
directed the investigation's pacing and priorities at each decision point
(what to test next, when to stop tuning and run the real thing, when to
demand a backtest instead of trusting accuracy alone) but did not write any
of the code or analysis herself.
