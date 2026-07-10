"""Baseline forecasts to contextualize the LSTM's own test metrics.

Neither baseline touches X or the model — both are computed purely from the
already time-ordered `y` (one-hot outcome) arrays. If the LSTM can't beat these, it
isn't learning anything useful from the engineered features; it's just an expensive
way to approximate a trivial rule.
"""

from __future__ import annotations

import numpy as np


def _accuracy(y_true_idx: np.ndarray, y_pred_idx: np.ndarray) -> float:
    return float(np.mean(y_true_idx == y_pred_idx))


def majority_class_baseline(y_train: np.ndarray, y_eval: np.ndarray) -> dict:
    """Always predict whichever class was most common in the training set — the
    simplest possible baseline: a "model" that ignores the input entirely.

    `correct` is a per-row boolean array aligned 1:1 with `y_eval` (and therefore
    with the LSTM's own predictions on the same test set) — enables a paired
    significance test (McNemar's) rather than just comparing aggregate accuracy.
    """
    train_idx = np.argmax(y_train, axis=1)
    majority_class = int(np.bincount(train_idx).argmax())

    eval_idx = np.argmax(y_eval, axis=1)
    predictions = np.full_like(eval_idx, majority_class)
    correct = eval_idx == predictions
    return {"accuracy": _accuracy(eval_idx, predictions), "correct": correct}


def persistence_baseline(y_eval: np.ndarray, exit_bar_offset: np.ndarray) -> dict:
    """Predict that this period's class is the same as the most recent PRIOR
    period's actual class -- but only once that prior period's label has actually
    RESOLVED, not just "the immediately preceding row regardless."

    This causal-validity requirement matters a lot here, and didn't used to be
    enforced: under triple-barrier labeling, a row's label isn't knowable until its
    barrier race actually finishes, which -- with max_holding_bars=24 on hourly
    bars -- takes a mean of ~15 bars (median 16; only ~7.6% resolve within 3 bars).
    An earlier version of this function predicted row i using row i-1's actual
    label unconditionally. Measured directly on a real EUR/USD H1 test split, 98.5%
    of rows had that "previous row"'s label still unresolved at the point it was
    being used -- i.e. it was borrowing up to 23 bars of future information. That
    version scored ~0.856; this causally-valid version, using only labels that had
    genuinely resolved by the time they'd be used, scores ~0.388 -- barely above
    the majority-class baseline, not the dominant baseline the old number implied.

    Algorithm: row j's own label becomes usable at bar `j + exit_bar_offset[j]`
    ("available_at"). For each row i, persist from the label of the largest row
    index j with available_at[j] <= i -- the most recent ENTRY (not necessarily the
    most recently-available information; these can genuinely differ when offsets
    aren't monotonic, e.g. an older entry that took a long time to resolve vs. a
    newer one that resolved fast -- "most recent entry" is the simpler, more
    standard generalization of lag-1 persistence, chosen deliberately). Computed as
    a running max over `available_at`-bucketed row indices, not a sequential
    forward scan -- a forward scan stalls behind any single slow-to-resolve early
    row even after later rows have resolved, since exit_bar_offset is not
    monotonic across rows.

    `y_eval`/`exit_bar_offset` must already be in chronological order (true of
    every Splits array in this pipeline). Rows with no resolved prior row yet
    (mostly the first several dozen rows, not just row 0) are excluded from
    scoring, the same spirit as excluding row 0 in the old version, just now a
    data-dependent count. `correct`/`scored` are both returned at FULL length
    (matching `y_eval`), replacing the old fixed "always N-1 rows" convention,
    since the excluded-row count is no longer constant: `scored[i]` is True iff row
    i had a resolved prior row to persist from; `correct[i]` is only meaningful
    where `scored[i]` is True. Align both `correct` and `scored` (as a boolean
    mask) against another model's full-length correctness array before comparing
    pairwise, rather than assuming a fixed row offset.
    """
    eval_idx = np.argmax(y_eval, axis=1)
    n = len(eval_idx)

    available_at = np.arange(n) + exit_bar_offset
    in_bounds = available_at < n  # a row whose own resolution falls beyond this
                                   # eval window can never be used as a causally
                                   # available prior label within it

    max_j_at = np.full(n, -1, dtype=int)
    valid_rows = np.nonzero(in_bounds)[0]
    # np.maximum.at (not plain fancy-index assignment) so that if two rows resolve
    # at the exact same bar, the larger (more recent) row index wins the bucket
    np.maximum.at(max_j_at, available_at[valid_rows], valid_rows)

    # running_max_j[i] = the largest j with available_at[j] <= i, for every i
    running_max_j = np.maximum.accumulate(max_j_at)
    scored = running_max_j >= 0

    # row index 0 is a VALID resolved row, so -1 (no resolved row yet) can't be
    # used to index eval_idx directly without risking silent negative-index
    # wraparound to eval_idx[-1] for unscored rows -- clamp to 0 (a dummy,
    # discarded value) and mask the result instead of trusting the sentinel
    safe_prior_idx = np.where(scored, running_max_j, 0)
    raw_correct = eval_idx[safe_prior_idx] == eval_idx
    correct = np.where(scored, raw_correct, False)

    if not scored.any():
        return {"accuracy": float("nan"), "correct": correct, "scored": scored}
    return {"accuracy": float(np.mean(correct[scored])), "correct": correct, "scored": scored}
