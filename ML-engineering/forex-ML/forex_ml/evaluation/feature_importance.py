"""Permutation importance for arbitrary fitted models, plus a horizontal bar plot
of the result.

Not tied to any specific model class -- unlike sklearn's own `.feature_importances_`,
permutation importance (shuffle one feature, or one named group of feature columns,
and see how much a scoring metric drops) works on anything with a `.predict()`, which
matters here specifically because `HistGradientBoostingClassifier` (used throughout
this project's GBT experiments) doesn't expose `.feature_importances_` at all.
"""

from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score


def compute_permutation_importance(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    feature_groups: dict[str, list[int]] | None = None,
    n_repeats: int = 5,
    random_state: int | None = None,
    scoring: Callable[[np.ndarray, np.ndarray], float] | str | None = None,
) -> pd.DataFrame:
    """Returns a DataFrame with columns [feature, importance_mean, importance_std],
    one row per feature (or per group, if `feature_groups` is given), unsorted.

    `feature_groups`, if given, maps a group name to the list of column indices in
    `X` that belong to it -- all of those columns are shuffled together, using the
    SAME row permutation, rather than independently. This is what you want when a
    single logical feature spans multiple columns (e.g. one named feature repeated
    across every lag of a flattened lookback window): shuffling each lag column
    independently would mostly just add noise within a row rather than actually
    removing that feature's information, understating its true importance.

    Without `feature_groups`, this is a thin wrapper around
    `sklearn.inspection.permutation_importance` (one column at a time, sklearn's
    own row-permutation logic) -- the general case for any estimator/data where
    each column already IS one distinct feature.
    """
    if feature_groups is None:
        result = permutation_importance(
            estimator, X, y, n_repeats=n_repeats, random_state=random_state, scoring=scoring,
        )
        return pd.DataFrame({
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        })

    if isinstance(scoring, str):
        raise TypeError("grouped permutation importance requires a callable scoring function, not a scorer name string")
    score_fn: Callable[[np.ndarray, np.ndarray], float] = accuracy_score if scoring is None else scoring
    rng = np.random.default_rng(random_state)
    baseline_score = score_fn(y, estimator.predict(X))

    rows = []
    for group_name, columns in feature_groups.items():
        drops = []
        for _ in range(n_repeats):
            perm = rng.permutation(len(X))
            X_perm = X.copy()
            X_perm[:, columns] = X[perm][:, columns]
            drops.append(baseline_score - score_fn(y, estimator.predict(X_perm)))
        rows.append({"feature": group_name, "importance_mean": np.mean(drops), "importance_std": np.std(drops)})
    return pd.DataFrame(rows)


def plot_feature_importances(
    importances: pd.DataFrame,
    top_n: int | None = None,
    title: str | None = None,
    metric_label: str = "Importance (mean score decrease when shuffled)",
    output_path: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Horizontal bar plot of `importances` (as returned by
    `compute_permutation_importance`), most important feature at the TOP and least
    important at the bottom. Bars are colored by sign: a positive mean (shuffling
    hurt performance -- the feature carries real signal) vs. zero-or-negative (shuffling
    didn't hurt, or even helped -- no real signal, or pure noise in this fold).

    Pass `top_n` to only plot the N most important features -- useful when
    `importances` covers hundreds/thousands of individually-named columns.
    """
    df = importances.sort_values("importance_mean", ascending=False)
    if top_n is not None:
        df = df.head(top_n)
    df = df.sort_values("importance_mean", ascending=True)  # ascending so barh puts the largest at the top

    if figsize is None:
        figsize = (8, max(2, 0.35 * len(df)))
    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(df))
    colors = ["#4C72B0" if v > 0 else "#B0B0B0" for v in df["importance_mean"]]
    xerr = df["importance_std"] if "importance_std" in df.columns else None
    ax.barh(y_pos, df["importance_mean"], xerr=xerr, color=colors, ecolor="black", capsize=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["feature"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(metric_label)
    if title:
        ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150)
    return fig
