# Our Heroine Retires the Stunt Double

Somewhere in the middle of an unrelated conversation about database schedulers
and swap rates, our heroine asked herself a question she really should have
asked months earlier: *does any of this actually work for short trades, or
just long ones?*

She went and checked. The honest answer was uncomfortable. Every prediction
her model made came in three flavors — go long, go short, stay flat — and the
"go short" flavor had never once been independently earned. It was a stunt
double. Whenever the long trade would have hit its stop-loss, the label
machinery shrugged and relabeled the whole row "short signal," on the theory
that a bad day for the bulls must be a good day for the bears. Nobody had ever
gone back and actually checked whether a short trade entered at that same
moment would have made a single cent.

It was time to give the short side its own stunts.

**How the Stunt Double Got the Gig**

The labeling scheme underneath all of this is called triple-barrier labeling,
and it's a nice idea: instead of asking "what's the price doing in exactly
four bars," which was already fired for being meaningless noise a few
chapters back, it watches a hypothetical trade forward bar by bar and asks
which of three things happens first — the price rallies far enough to clear a
profit-take line, it drops far enough to hit a stop-loss line, or neither
happens before time runs out. Whichever fires first, net of real spread and
swap cost, becomes the label: a win, a loss, or a timeout.

The catch, sitting quietly in the module's own comments since the day it was
written, was the word "long." The whole race was run assuming you'd bought
the pair. If that hypothetical long trade's stop-loss fired, the label became
`-1`. Somewhere along the way, `-1` picked up a second job title: "short
signal." Which would be a perfectly reasonable shortcut, if a long losing
money and a short making money were the same event wearing two hats. They are
not.

**Why a Mirror Isn't Always a Mirror**

Here's the part worth sitting with, because it's the entire reason this
chapter exists and it isn't obvious until you actually do the arithmetic.

Trading costs money to enter and exit — spread, and if you hold overnight,
swap. That cost gets subtracted from whatever the trade is doing. For a long
position, subtracting a cost makes losses arrive *faster*: the price only has
to fall most of the way to the stop-loss line before the cost tips it the
rest of the way over. For a short position, subtracting that same cost makes
*wins* arrive slower: the price has to fall further than the raw profit-take
line, because the cost is eating into the gain before the finish line even
gets checked.

Same direction of price movement. Two completely different amounts of price
movement required, depending which side of the trade is asking. A long
stopping out is a *low bar*. A short actually clearing its own profit,
honestly priced, is a *higher* one. Treating the first as proof of the
second was quietly handing the short side a victory it hadn't earned.

A concrete version, using numbers close to this project's own thresholds:
profit-take and stop-loss both set at three-tenths of a percent, a small
round-trip cost of two-hundredths of a percent, and a modest swap charge on
each side. A price drop of exactly three-tenths of a percent is enough, once
cost is added in, to stop the long out — table stakes, the bar was already
that close. That same three-tenths of a percent drop, checked honestly
against the short's *own* economics, falls a bit short of clearing its
own profit-take line once its own cost is subtracted. The short's own
finish line sits a little further out — closer to five-tenths of a percent,
in this example — before it can honestly call itself a win. Anything in that
gap used to get filed as "short signal." Now it correctly gets filed as
"nothing happened, don't bother."

**Two Races, Not One Race Wearing a Costume**

The fix was to stop pretending one race could stand in for both and just run
both races, for real, on every single row. One hypothetical long trade, one
hypothetical short trade, both starting from the exact same entry, both
walking forward bar by bar, each checked against its *own* cost — including
its own swap rate, since a long and a short position on the same pair are
very often charged completely different overnight rates, sometimes even in
opposite directions.

Whichever race's profit-take line gets crossed first wins the row. If only
the long's does, the label is "go long." If only the short's does, the label
is "go short." If neither ever manages it — one or both raced all the way to
their own stop-loss, or simply ran out of time — the honest answer is
"nothing here," not a consolation prize for whichever side merely lost less
badly.

There was one delightfully weird wrinkle in the design review of all this:
could both races possibly cross their own finish line on the exact same
tick? On paper, with sane inputs, no — a long winning and a short winning at
the same instant would require the price to have simultaneously gone up a
lot and down a lot, which price does not do. But "on paper, with sane
inputs" is exactly the kind of sentence that should make anyone nervous, so
rather than trust the assumption, the code now also **checks** it: negative
spread values get rejected outright (a nonsensical input that would have
broken the "impossible" argument), and there's an explicit, tested rule for
who wins if the impossible ever does happen anyway. Nobody had to lose sleep
over which rule was fairer — long simply keeps its seniority in a tie, and a
test file now proves the rule fires instead of assuming the situation
politely stays hypothetical forever.

**What Changes, and What Doesn't**

The shape of the model's output didn't change at all — three classes, same
short/flat/long convention the rest of the pipeline and the backtest already
expected, so nothing downstream needed to be told the news. What changed is
what "short" actually *means*: it now requires the same standard of proof a
long signal always had to meet, instead of getting waved through on a
technicality.

One honest consequence, flagged rather than buried: the "flat" class is
expected to get more crowded. A short signal is now a genuinely harder bar to
clear than it used to be, so some rows that used to get filed under "short"
will correctly land under "nothing here" instead. That's the fix doing its
job, not a step backward — a model that's honest about having less to say
some of the time is more useful than one that was confidently wrong about
half its short calls.

And no, the actual training run has not been restarted yet, on purpose. The
target the model would be learning just changed what it means, and that's a
big enough shift to deserve its own deliberate green light rather than
getting swept up as a footnote to a labeling fix.

**Where Things Stand**

The short side of this model finally has to earn its own paycheck. No more
riding in on the long trade's bad day and calling it a win. Both directions
now get raced, honestly, on their own terms, with their own costs — and the
one edge case weird enough to make a reasonable person nervous got checked
instead of assumed away.

**AI Use Statement:** Claude Code designed and implemented the bidirectional
triple-barrier labeling redesign described in this post (the two-race
long/short labeling logic, the swap-cost plumbing changes, the tie-break and
input-validation rules, and the accompanying tests across both the forex-ML
and forex-strategy repositories), including a second independent review pass
that specifically stress-tested the "simultaneous win" edge case and the
historical-compatibility handling of the config-signature hashing. It also
wrote this post's prose. The author identified the original long-only
limitation, requested the redesign, and reviewed the resulting change before
it shipped.
