# Our Heroine vs. the Exploding Model

With real data finally flowing and a GPU finally cooperating (a saga involving a
missing environment variable, best left for the footnotes), our heroine was ready for
the moment this whole project had been building toward: actually training the LSTM,
for real, at the configuration the pipeline had been carrying around unquestioned
since the original notebooks — `n_back=200`, five stacked 300-unit LSTM layers, the
whole production-sized apparatus. She hit go, watched the first epoch crawl through
its 3,129 batches at a very respectable clip, and settled in.

Batch 366 of 3,129, loss went to `nan`. And stayed there. Forever.

**A Diagnosis, Not a Panic**

NaN loss is the machine-learning equivalent of a patient flatlining — dramatic,
alarming, and not actually informative on its own about *why*. Our heroine's first
instinct was the responsible one: check the data before blaming the model. She loaded
the exact tensors that had gone into that batch and checked for `NaN`, checked for
`Inf`, checked the value ranges. Clean. Every input was a real, finite number. Whatever
had gone wrong, it wasn't corrupted data quietly poisoning the well — it was something
the *model itself* was doing to otherwise perfectly good inputs.

That left one very well-known suspect: exploding gradients. Five stacked LSTM layers,
each one unrolling across 200 timesteps of backpropagation-through-time, is a lot of
multiplicative depth for a gradient to compound across — and a quick check of
`compile_model()` confirmed the optimizer had no gradient clipping configured
whatsoever. Not a regression, either: the *original* `lstm.py`, dug up from git
history, never had it either. This particular landmine had been sitting in the code
since the very beginning, patiently waiting for someone to actually train at the
pipeline's real configured depth on real, full-scale data — which, as it happens,
nobody ever had, until this exact moment.

**The Fix That Almost Worked**

`clipnorm=1.0` on the optimizer, a new configurable `gradient_clip_norm` in the
pipeline's params, one clean retry — and the model trained beautifully. Epoch after
epoch of steadily improving validation loss, no drama, right up through epoch 12.

Then epoch 13, batch 265: `nan` again.

This is the part of the story where a less careful engineer declares victory too
early. The fix *had* helped — the explosion that used to happen at batch 366 of epoch
1 now took twelve entire clean epochs to resurface — but "later" is not the same
thing as "gone." Something about gradient clipping's actual mechanism wasn't covering
the whole problem, and our heroine's job was to figure out what.

**What Clipping a Gradient Cannot Fix**

The answer, once she thought about it properly, was almost embarrassingly clean:
gradient clipping operates on the *backward* pass. It takes a gradient that has
already been computed and rescales it if it's too large. But if the *forward* pass
itself — the actual arithmetic of running data through the network — produces
infinity, there is no gradient left to clip. Infinity minus infinity, or infinity
times zero, is `NaN`, full stop, and clipping a `NaN` still gives you a `NaN`.

So: what could make a forward pass overflow? A sufficiently extreme input value,
propagating and compounding across 200 recurrent timesteps, is exactly the kind of
thing that does it — and this pipeline's price-based features are *known*, from
several chapters back, to have serious tails. `return`'s kurtosis was measured at
roughly 17 in an earlier investigation — a textbook fat-tailed distribution, not a
gentle bell curve. A quick scan of the actual normalized training tensors confirmed
it in concrete terms: thousands of individual values, in every non-cyclical feature,
sitting beyond ten standard deviations from the mean. Rare on any single day, but with
tens of thousands of training windows each 200 bars long, "rare" adds up to "present
in nearly every batch eventually."

**Clipping the Input, Not Just the Gradient**

The fix this time lives at the front door instead of the back one: a new `ClipInputs`
layer, inserted as the very first layer of the model, clamping every input value to
`[-10, 10]` before it ever reaches the first LSTM cell. Gradient clipping and input
clipping are not the same tool wearing two hats — they guard against two different
failure points, and this pipeline needed both. `ClipInputs` is deliberately a proper
Keras `Layer` subclass rather than the more casual `Lambda` shortcut, specifically so
it survives being saved and reloaded — this pipeline's checkpointing depends on
`keras.models.load_model()` faithfully reconstructing the exact architecture it saved,
and a bare `Lambda` is a well-known way to make that reconstruction quietly fail later.

**An Honest Postscript**

With both fixes in place, training finally ran clean start to finish, thirteen
epochs, early stopping doing its job, best weights restored. And the headline result
was... a little humbling. The trained model's test accuracy came in well *behind* a
trivially simple baseline — "assume this period looks like the last one" — which is
exactly the kind of result a less rigorous project would be tempted to quietly not
mention. Digging into *why* turned out to be its own small, worthwhile investigation:
part of that baseline's apparent strength is a structural artifact of how the
prediction target overlaps itself bar-to-bar, not evidence of some genuinely
persistent market regime — but a confusion matrix also showed the model doing
distinctly worse than chance at telling the "big gain" class apart from the "big
drop" class, which is a real, substantive finding about how hard this particular
target is to predict from this particular feature set, not a bug to be fixed away.
That's a question for a future chapter, not this one. Today's job was making sure the
model could finish a training run at all without quietly destroying itself — and now
it can.

**AI Use Statement:** Claude Code diagnosed both training-instability bugs described
in this post (the missing gradient clipping and the forward-pass overflow from
unclipped extreme inputs), implemented both fixes (`gradient_clip_norm` and the
`ClipInputs` layer, including its save/load serialization test), and wrote this post's
prose itself, across an extended collaborative session with the author, who directed
the investigation and reviewed each finding as it came in.
