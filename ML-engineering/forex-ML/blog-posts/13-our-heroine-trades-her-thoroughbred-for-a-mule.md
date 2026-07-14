# Our Heroine Trades Her Thoroughbred for a Mule

Last chapter ended on an uncomfortable note. A model had aced its exam — beaten
the honest baselines, cleared a real significance test — and then walked
straight into a cost-aware backtest and lost money doing it. Worse, when our
heroine went looking for *why*, she found the model wasn't quietly unlucky. It
was unstable: the exact same architecture, trained on the exact same data,
differing only in which random number generator seed happened to be sitting in
the config, would sometimes learn something real and would sometimes just give
up and guess one class over and over. Before chasing profitability any further,
she'd promised herself a proper look under the hood.

**The Architecture's Guilty Secret**

The model was a stack of LSTM layers reading two hundred hours of history,
followed by a small classifier head. Digging into exactly how those two pieces
connected turned up something nobody had questioned since the very first draft
of this project: every single recurrent layer — including the very last one —
was configured to hand forward its *entire* sequence of hidden states, not just
its final, settled-down summary of everything it had seen. That meant the last
layer wasn't handing the classifier one clean, 300-number summary of "here's
what I learned from the last 200 hours." It was handing over all 200 hours'
worth of hidden states at once, flattened into a single 60,000-number list,
which then got squeezed down to a mere seven numbers before the model made its
call.

That's an enormous, extremely repetitive pile of numbers to funnel through a
narrow doorway — adjacent hours' hidden states barely differ from each other,
so most of those 60,000 numbers were saying nearly the same thing. A pile that
redundant, feeding a doorway that narrow, is exactly the kind of setup where
tiny differences in starting conditions can send training down wildly
different paths. It was a genuinely well-reasoned suspect.

While in there, a second, unrelated problem turned up: the very last layer —
the one whose job is to output "how confident am I in short, flat, or long" —
had a regularization term attached directly to its own output. Not to its
weights. To the actual probabilities it hands back. That's a straightforward
mistake — it means the model was being penalized, during training, every single
time it tried to sound confident about anything. And it lined up perfectly with
something already on record from a previous chapter: this model had never once,
across an entire test set, been more than fifty percent sure about its own call.

**A Confound, Caught Just in Time**

Before getting attached to either fix, it was worth double-checking the very
evidence that started this whole chapter — two real training runs, identical
except for one random seed, one landing at a strong result and the other
collapsing into guessing "long" on ninety-nine percent of rows. A closer look at
the actual training logs found the collapsed run hadn't calmly converged to a
bad answer. It had blown up to nonsense values within its very first pass
through the data and only survived because of the safety-net checkpoint built
in a few chapters back. That's a different, already-understood failure mode —
not proof that different seeds settle into different bad habits.

That could have been the end of the investigation right there — an
embarrassing "well, never mind" — except a free check was sitting right there
for the taking: every training run this project had logged all session already
had its collapse status and its blow-up status recorded, no retraining
required. Pulling that history for all thirty-three runs told a clearer story:
only two of them had collapsed for that same early-blow-up reason. Eleven more
had collapsed while training completely normally — no crash, no shortcut, one
of them a full twenty-eight epochs in. Genuine collapse, real and common,
independent of the blow-up problem. The original hunch survived its own
fact-check.

**Testing the Fix Properly**

With the hunch still standing, it was time to actually test it — and given the
chapter's own opening lesson, testing it on a single run wasn't going to cut it
this time. Five different random seeds, run twice each: once with the
architecture as originally inherited, once with the last layer fixed to hand
over its proper single summary instead of the whole 60,000-number pile.

The fixed version lost. Decisively. Every single one of its five runs collapsed
to guessing one class almost exclusively — compared to two out of five for the
original. Its blow-up rate nearly quadrupled. Its average accuracy dropped
below where a trivial guess-the-most-common-class rule would land. A carefully
reasoned hypothesis, tested honestly, and refuted.

The likely explanation is almost funny: that huge, wasteful, redundant pile of
60,000 mostly-repetitive numbers may not have been pure waste after all. It may
have been acting as an accidental cushion — a landscape with a lot of
mediocre-but-not-catastrophic places to land, rather than a small, sharp one
with fewer places to land and more of them bad. Tidying up the architecture
made it leaner and, it turns out, considerably more fragile. That fix got
reverted.

**A Correct Fix That Didn't Actually Do Anything**

The second suspect — the regularizer quietly punishing the model for sounding
confident — got tested on its own, separated from the reverted architecture
change. Across the same five seeds, it made no measurable difference at all.
Same collapse rate. Nearly identical average confidence. It stayed exactly as
theoretically wrong as it always was, but it wasn't the explanation for
anything observed so far. It got fixed anyway, on principle — there's no
version of "penalize the model for its own answers" that's defensible, whether
or not it turns out to matter — and the investigation moved on.

**Two More Swings, Two More Misses**

Two more candidates got a fair test. The first: weighting rarer classes more
heavily during training, so the model can't win by just leaning on whichever
outcome happened to be a little more common in a given stretch of history. That
one was a genuine trade-off rather than a clean loss — it cut the collapse rate
in half, but it also dragged average accuracy down below the trivial baseline
and nearly doubled the blow-up rate. Making rare mistakes cost more apparently
also means the gradient signal for them gets sharper and more aggressive,
which — on an architecture already known to be temperamental — bought fewer
collapses at the price of more explosions. Not obviously a win.

The second: a stray discovery that one of the model's regularization strengths
had drifted tenfold stronger than the value in the very first draft of this
project, with no record of anyone deciding to change it. Reverting it back to
that original, gentler value seemed like the safe, conservative move. It was
the worst result of the entire investigation — every single one of five seeds
blew up and collapsed. Whatever that stronger setting was doing, it wasn't an
accident worth undoing; it was actively holding a fragile system together, and
loosening it let the whole thing come apart.

**Time to Ask a Bigger Question**

Four honest attempts, four architecture-side dead ends. At that point the more
useful question stopped being "which dial do I turn" and became "is this the
right kind of engine at all." A few directions got weighed out loud: a much
shorter memory window, since two hundred hours of unrolled history is itself a
lot of what makes this thing so temperamental; a training target that measures
money instead of a class label, since a chapter ago the class label had already
proven it doesn't reliably mean money; and, cheapest and fastest to actually
check, trying an entirely different, famously boring kind of model on the exact
same inputs.

That last one won out, for one simple reason: it would immediately answer
whether the whole session had been fighting a broken engine, or fighting a
genuine absence of anything to find in the first place.

**The Mule Shows Up**

A gradient-boosted tree model — no unrolled memory, no exploding-gradient
mechanism to speak of, nothing resembling any of the last few chapters' fragile
machinery — got handed the exact same engineered features and the exact same
labels. Three different random seeds, same as any fair test deserves.

It was boringly, wonderfully consistent. Every seed landed within a hair of the
same accuracy. Every seed beat both the majority-guess baseline and the
honestly-recalculated persistence baseline, comfortably and repeatably. No
run ever collapsed into guessing one thing. This was the cleanest, most
reassuring result of the entire investigation: there is real, usable structure
sitting in this data. It was never absent. It was just too much for a
temperamental thoroughbred to reliably reach.

**Sequencing the Next Question Properly**

A fork appeared here too: keep polishing this new, steadier model, or go back
and reconsider the whole training target the way the "shorter memory" and
"predict money instead of a label" ideas had suggested? Rather than guess, the
honest move was to let one cheap, already-available result decide: run this
new model through the same real, cost-aware backtest the thoroughbred had
already failed, and let *that* answer determine which fork actually mattered.

**A Promising Number, Caught Before It Went to Print**

The backtest turned up something genuinely new: at one particular confidence
threshold, filtering down to only the model's more convinced calls, the win
rate climbed above fifty percent for the first time all session, and the net
result — after real trading costs — came out slightly positive. The very
first positive number this entire investigation had produced.

It didn't get celebrated. It got checked. A quick significance test on that
result said it wasn't yet distinguishable from a lucky coin — not wrong to be
curious about, but nowhere near strong enough to trust on its own.

**The Number That Didn't Survive Its Sequel**

So it got a sequel: the same model, freshly retrained, run through the same
backtest, across five separate, non-overlapping stretches of real market
history rather than just the one. The promising pattern showed up clearly in
exactly one of the five. Pooled across all five together, the win rate never
climbed meaningfully above fifty-fifty at any confidence level, and if anything
drifted the wrong way as the bar for confidence went up. The one encouraging
number from a chapter ago turned out to be exactly what its own significance
test had warned it might be — a coincidence, not a discovery.

**Where Things Stand**

A genuinely busy chapter, and an honest one to sum up. A well-reasoned
architecture fix got tried and disproven rather than assumed. A real
correctness bug got fixed regardless of whether it moved any number. Two more
honest attempts came back as real trade-offs or outright losses instead of
quiet wins. And the biggest result of the day — swapping the whole model
family for something sturdier — proved that real signal exists in this data,
even though it hasn't yet produced a trading edge that survives being asked
twice.

That's not nothing. The thoroughbred is still fast and still occasionally
brilliant, but it's proven, repeatedly now, that it can't be trusted to show up
sober. The mule shows up every time — it just hasn't won a race yet. Knowing
which of those two problems you actually have is worth more than another month
of tuning the wrong one.

**AI Use Statement:** Claude Code ran this entire investigation described in
this post — the architecture and regularizer analysis, the seed-confound audit,
the return_sequences/regularizer/class-weight/L1-regularization comparisons,
setting up and evaluating the gradient-boosted-tree model, the initial and
multi-window backtests, and the significance testing throughout — and wrote
this post's prose. The author set the direction at each decision point
(ordering the architecture investigation before further tuning, choosing to
test a fundamentally different model class, insisting on the multi-window
validation before trusting the promising backtest result) but did not write
any of the code or run any of the analysis herself.
