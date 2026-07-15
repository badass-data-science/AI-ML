"""ROC and calibration plots for a multiclass classifier's test-set `predict_proba`
output.

Both are one-vs-rest per class (this project's models are always 3-class:
short/flat/long), plotted together on one Axes per figure. ROC answers "how well
does this class's predicted probability rank true positives above true negatives" --
the literal, standard question. Calibration answers a different, arguably more
consequential question for this project specifically: `forex_strategy.backtest`'s
`predicted_classes_to_positions` thresholds trades by `min_confidence` directly
against `predict_proba`'s output, which only makes sense if a predicted probability
of e.g. 0.55 really does correspond to being right about 55% of the time -- a model
can have excellent ROC/AUC (good at ranking) while being badly miscalibrated (bad at
producing trustworthy confidence thresholds), so this project's confidence-threshold
trading rule specifically needs the calibration view, not just ROC.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, roc_curve


def plot_multiclass_roc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
    title: str | None = None,
    output_path: str | None = None,
    figsize: tuple[float, float] = (6, 6),
) -> Figure:
    """`y_true`: 1D array of class indices. `y_proba`: (n_samples, n_classes) array,
    e.g. straight from `.predict_proba()`. One one-vs-rest ROC curve per class,
    each labeled with its AUC, plus the diagonal no-skill reference line."""
    fig, ax = plt.subplots(figsize=figsize)
    for class_idx, class_name in enumerate(class_names):
        y_true_binary = (y_true == class_idx).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_proba[:, class_idx])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    if title:
        ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150)
    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
    n_bins: int = 10,
    title: str | None = None,
    output_path: str | None = None,
    figsize: tuple[float, float] = (6, 6),
) -> Figure:
    """Reliability diagram: one one-vs-rest calibration curve per class -- x-axis is
    the mean predicted probability within a bin, y-axis is that bin's actual
    empirical frequency of the class. A well-calibrated model tracks the diagonal;
    a curve consistently below it means predicted confidence overstates real
    accuracy at that confidence level (the failure mode that would most directly
    undermine `min_confidence` trade filtering)."""
    fig, ax = plt.subplots(figsize=figsize)
    for class_idx, class_name in enumerate(class_names):
        y_true_binary = (y_true == class_idx).astype(int)
        frac_positive, mean_predicted = calibration_curve(
            y_true_binary, y_proba[:, class_idx], n_bins=n_bins, strategy="uniform",
        )
        ax.plot(mean_predicted, frac_positive, marker="o", label=class_name)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfectly calibrated")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical frequency")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150)
    return fig
