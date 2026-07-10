from __future__ import annotations

import numpy as np

from forex_ml.evaluation.baselines import majority_class_baseline, persistence_baseline


def _one_hot(class_indices: list[int], n_classes: int = 3) -> np.ndarray:
    return np.eye(n_classes)[class_indices]


def test_majority_class_baseline_predicts_most_common_train_class():
    y_train = _one_hot([0, 0, 0, 1, 2])  # class 0 is most common
    y_eval = _one_hot([0, 1, 0, 2])  # 2 of 4 rows are actually class 0

    result = majority_class_baseline(y_train, y_eval)
    assert result["accuracy"] == 0.5


def test_persistence_baseline_perfect_when_class_never_changes():
    y_eval = _one_hot([0, 0, 0, 0])
    offsets = np.array([1, 1, 1, 1])  # every row resolves immediately
    result = persistence_baseline(y_eval, offsets)
    assert result["accuracy"] == 1.0


def test_persistence_baseline_zero_when_class_always_alternates():
    y_eval = _one_hot([0, 1, 0, 1])
    offsets = np.array([1, 1, 1, 1])
    result = persistence_baseline(y_eval, offsets)
    assert result["accuracy"] == 0.0


def test_persistence_baseline_reduces_to_classic_lag_one_when_every_offset_is_one():
    """With every row resolving in exactly 1 bar, this is exactly the old
    unconditional lag-1 persistence baseline -- a continuity check against the
    pre-fix behavior for the simplest case."""
    y_eval = _one_hot([1, 0, 0, 0])
    offsets = np.array([1, 1, 1, 1])
    # pred[i] = actual[i-1] for i in 1..3: [1,0,0] vs actual [0,0,0] -> 2 of 3 correct
    result = persistence_baseline(y_eval, offsets)
    assert result["accuracy"] == 2 / 3


def test_persistence_baseline_returns_nan_for_single_row():
    y_eval = _one_hot([0])
    offsets = np.array([1])
    result = persistence_baseline(y_eval, offsets)
    assert np.isnan(result["accuracy"])


def test_persistence_baseline_excludes_rows_whose_prior_label_has_not_resolved_yet():
    """The core fix. Row 0 takes 10 bars to resolve (never resolves within this
    5-row window at all); rows 1-3 each resolve in 1 bar. The naive (pre-fix)
    baseline would have scored row 1 by persisting row 0's actual label
    regardless -- comparing class 0 against row 1's actual class 1, a WRONG
    prediction made using information (row 0's true outcome) that wasn't actually
    knowable yet. The fix must not score row 1 at all, not merely avoid crediting
    it: there simply isn't a resolved prior label to persist from."""
    y_eval = _one_hot([0, 1, 1, 1, 1])
    offsets = np.array([10, 1, 1, 1, 1])

    result = persistence_baseline(y_eval, offsets)

    np.testing.assert_array_equal(result["scored"], [False, False, True, True, True])
    assert result["accuracy"] == 1.0  # all 3 scoreable rows (2,3,4) persist correctly


def test_persistence_baseline_persists_from_the_most_recent_resolved_entry_not_the_freshest_information():
    """When exit_bar_offset isn't monotonic, "most recent entry with a resolved
    label" and "whichever label became available most recently" can genuinely
    differ. Row 4 (offset=1) resolves at bar 5 -- one bar stale by bar 6. Row 1
    (offset=5) resolves exactly at bar 6 -- freshest possible information, but an
    older entry. This pins down the deliberate choice: persist from row 4 (the
    more recent ENTRY), not row 1 (the more recently AVAILABLE information).
    Constructed so the two choices predict different classes at row 6, so the
    correct/incorrect outcome itself proves which one the code used."""
    # row:      0  1  2  3  4  5  6
    offsets = np.array([1, 5, 1, 1, 1, 2, 1])
    # row 4 == row 6's actual class (so "most recent entry" predicts correctly);
    # row 1 is a different class (so "freshest information" would predict wrong)
    y_eval = _one_hot([0, 1, 0, 0, 2, 0, 2])

    result = persistence_baseline(y_eval, offsets)

    assert result["scored"][6]
    assert result["correct"][6]  # would be False if row 1 (freshest info) were used instead


def test_persistence_baseline_scores_the_exact_resolution_bar_not_just_strictly_after():
    """available_at[j] <= i (not < i) is the correct boundary: row i's own entry
    is priced off the same bar i that row j's race would resolve against if
    available_at[j] == i exactly, so it's contemporaneous information, not a
    future leak. Row 0 resolves at exactly bar 1 -- must be usable at row 1."""
    y_eval = _one_hot([1, 0])
    offsets = np.array([1, 3])  # row 0 resolves at exactly bar 0+1=1

    result = persistence_baseline(y_eval, offsets)

    assert result["scored"][1]
    assert result["correct"][1] == (1 == 0)  # persisted row 0's class (1) vs row 1's actual (0)


def test_persistence_baseline_never_counts_an_unscored_row_as_correct_even_if_coincidentally_matching():
    """Regression guard for the unscored-row sentinel: if a row has no resolved
    prior label, it must read as not-correct regardless of what value happens to
    sit at whatever placeholder index an unmasked implementation might
    accidentally read from. All three rows here are the same class, so a naive
    unmasked comparison would spuriously read as "correct" -- the mask must
    override that."""
    y_eval = _one_hot([1, 1, 1])
    offsets = np.array([10, 10, 10])  # nothing resolves within this 3-row window

    result = persistence_baseline(y_eval, offsets)

    assert not result["scored"][1]
    assert not result["correct"][1]
    assert not result["scored"][2]
    assert not result["correct"][2]


def test_persistence_baseline_returns_nan_when_nothing_resolves_in_window():
    y_eval = _one_hot([1, 1, 1])
    offsets = np.array([10, 10, 10])
    result = persistence_baseline(y_eval, offsets)
    assert np.isnan(result["accuracy"])
