from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from forex_ml.evaluation.feature_importance import compute_permutation_importance, plot_feature_importances


def _fit_informative_vs_noise_classifier(rng, n_rows: int = 400):
    """Column 0 fully determines the label; columns 1-2 are pure noise -- any
    reasonable importance measure should rank column 0 far above the others."""
    X = rng.normal(size=(n_rows, 3))
    y = (X[:, 0] > 0).astype(int)
    clf = HistGradientBoostingClassifier(random_state=0).fit(X, y)
    return clf, X, y


def test_compute_permutation_importance_ranks_informative_feature_higher():
    rng = np.random.default_rng(0)
    clf, X, y = _fit_informative_vs_noise_classifier(rng)

    result = compute_permutation_importance(
        clf, X, y, feature_names=["informative", "noise_1", "noise_2"], random_state=0,
    )

    informative_row = result[result["feature"] == "informative"].iloc[0]
    noise_rows = result[result["feature"] != "informative"]
    assert informative_row["importance_mean"] > noise_rows["importance_mean"].max()


def test_compute_permutation_importance_grouped_shuffles_columns_jointly():
    rng = np.random.default_rng(0)
    n_rows = 400
    # Column 0 is informative and duplicated across a 3-column "lag" group; columns
    # 3-4 are pure noise, grouped separately.
    signal = rng.normal(size=n_rows)
    X = np.column_stack([signal, signal, signal, rng.normal(size=n_rows), rng.normal(size=n_rows)])
    y = (signal > 0).astype(int)
    clf = HistGradientBoostingClassifier(random_state=0).fit(X, y)

    result = compute_permutation_importance(
        clf, X, y, feature_names=[], feature_groups={"informative_group": [0, 1, 2], "noise_group": [3, 4]},
        random_state=0,
    )

    informative = result[result["feature"] == "informative_group"].iloc[0]["importance_mean"]
    noise = result[result["feature"] == "noise_group"].iloc[0]["importance_mean"]
    assert informative > noise


def test_plot_feature_importances_puts_most_important_at_top():
    import pandas as pd

    importances = pd.DataFrame({
        "feature": ["low", "high", "medium"],
        "importance_mean": [0.01, 0.30, 0.15],
        "importance_std": [0.002, 0.01, 0.005],
    })

    fig = plot_feature_importances(importances)
    ax = fig.axes[0]
    labels = [label.get_text() for label in ax.get_yticklabels()]

    assert labels[-1] == "high"  # highest y-position (top of the plot) is the most important
    assert labels[0] == "low"  # lowest y-position (bottom of the plot) is the least important
