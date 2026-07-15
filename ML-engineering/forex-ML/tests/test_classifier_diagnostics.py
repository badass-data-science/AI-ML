from __future__ import annotations

import numpy as np

from forex_ml.evaluation.classifier_diagnostics import plot_calibration_curve, plot_multiclass_roc

CLASS_NAMES = ["short", "flat", "long"]


def test_plot_multiclass_roc_gives_perfect_classifier_auc_of_one():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=300)
    # Predicted probability puts (almost) all mass on the true class -- a
    # near-perfect ranker should score AUC very close to 1.0 for every class.
    y_proba = np.full((300, 3), 0.01)
    y_proba[np.arange(300), y_true] = 0.98

    fig = plot_multiclass_roc(y_true, y_proba, CLASS_NAMES)
    ax = fig.axes[0]
    legend_text = ax.get_legend().get_texts()
    aucs = [float(t.get_text().split("AUC=")[1].rstrip(")")) for t in legend_text if "AUC=" in t.get_text()]

    assert len(aucs) == 3
    assert all(a > 0.99 for a in aucs)


def test_plot_multiclass_roc_gives_random_classifier_auc_near_half():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=2000)
    y_proba = rng.dirichlet(alpha=[1, 1, 1], size=2000)  # independent of y_true

    fig = plot_multiclass_roc(y_true, y_proba, CLASS_NAMES)
    ax = fig.axes[0]
    legend_text = ax.get_legend().get_texts()
    aucs = [float(t.get_text().split("AUC=")[1].rstrip(")")) for t in legend_text if "AUC=" in t.get_text()]

    assert all(0.4 < a < 0.6 for a in aucs)


def test_plot_calibration_curve_tracks_diagonal_when_well_calibrated():
    rng = np.random.default_rng(0)
    n = 5000
    # Predicted probability for the "long" class is drawn uniformly, and the true
    # label is generated so that P(long) really does equal that predicted value --
    # a textbook well-calibrated setup.
    p_long = rng.uniform(0, 1, size=n)
    is_long = rng.uniform(0, 1, size=n) < p_long
    y_true = np.where(is_long, 2, 0)
    y_proba = np.zeros((n, 3))
    y_proba[:, 2] = p_long
    y_proba[:, 0] = 1 - p_long

    fig = plot_calibration_curve(y_true, y_proba, CLASS_NAMES, n_bins=10)
    ax = fig.axes[0]
    long_line = next(line for line, label in zip(ax.get_lines(), CLASS_NAMES + ["_perfectly_calibrated_unused"])
                      if label == "long")
    mean_predicted, frac_positive = long_line.get_xdata(), long_line.get_ydata()

    assert np.allclose(mean_predicted, frac_positive, atol=0.1)
