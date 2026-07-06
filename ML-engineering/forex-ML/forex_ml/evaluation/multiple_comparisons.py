"""Multiple-comparisons-aware reporting across (instrument, granularity) pairs.

With N independent pairs each getting their own "does the LSTM beat the baseline?"
test, some pair will look significant by chance even if none has real signal — the
same failure mode Lopez de Prado's deflated Sharpe ratio targets for backtest
selection. This module:

  1. Runs McNemar's test per pair — the correct paired test for comparing two
     classifiers evaluated on the SAME test rows (uses only the discordant rows,
     where exactly one model was right; rows where both agree carry no information
     about which is better).
  2. Applies a Benjamini-Hochberg FDR correction across all pairs, so the number of
     "significant" results as a WHOLE stays honest instead of each pair being judged
     against a raw, uncorrected alpha.
"""

from __future__ import annotations

import mlflow
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests


def mcnemar_p_value(model_a_correct: np.ndarray, model_b_correct: np.ndarray) -> float:
    """p-value from McNemar's test comparing two classifiers' correctness on the
    same test rows. Requires both arrays aligned 1:1 to the same rows — see the
    persistence_baseline docstring for why its `correct` array needs an extra
    alignment step (it's one row shorter) before comparing against another model's
    full-length array.
    """
    if len(model_a_correct) != len(model_b_correct):
        raise ValueError(
            f"model_a_correct (len {len(model_a_correct)}) and model_b_correct "
            f"(len {len(model_b_correct)}) must be aligned to the same rows"
        )
    a_right_b_wrong = int(np.sum(model_a_correct & ~model_b_correct))
    a_wrong_b_right = int(np.sum(~model_a_correct & model_b_correct))
    both_right = int(np.sum(model_a_correct & model_b_correct))
    both_wrong = int(np.sum(~model_a_correct & ~model_b_correct))

    table = [[both_right, a_right_b_wrong], [a_wrong_b_right, both_wrong]]
    # exact binomial test below ~25 discordant pairs, chi-square with continuity
    # correction above — the standard rule of thumb for McNemar's test.
    result = mcnemar(table, exact=(a_right_b_wrong + a_wrong_b_right) < 25)
    return float(result.pvalue)


def benjamini_hochberg_report(pair_p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Applies BH FDR correction across all pair p-values. Returns, per pair, the
    raw p-value, the BH-adjusted p-value, and whether it's significant AFTER
    correction — the number that actually matters when you're looking across many
    pairs, not the raw per-pair p-value."""
    pairs = list(pair_p_values.keys())
    p_values = [pair_p_values[pair] for pair in pairs]
    reject, p_adjusted, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
    return {
        pair: {
            "p_value": p_values[i],
            "p_adjusted": float(p_adjusted[i]),
            "significant_after_correction": bool(reject[i]),
        }
        for i, pair in enumerate(pairs)
    }


def report_across_pairs(
    tracking_uri: str,
    experiment_name: str,
    baseline: str = "majority",
    alpha: float = 0.05,
) -> dict[str, dict]:
    """Pulls every run in `experiment_name`, downloads each run's predictions.npz
    artifact (saved by forex_ml.training.train.train_and_evaluate), runs McNemar's
    test per pair (LSTM vs the chosen baseline), and BH-corrects across all of them.

    If a pair has been trained more than once (expected on a single local GPU, where
    pairs get retrained individually over time as hyperparameters are refined rather
    than all trained together), only the MOST RECENT run for that pair is used —
    runs are pulled ordered by start_time descending and older duplicates for an
    already-seen pair are skipped.
    """
    if baseline not in ("majority", "persistence"):
        raise ValueError(f"baseline must be 'majority' or 'persistence', got {baseline!r}")

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"No MLflow experiment named {experiment_name!r}")

    pair_p_values: dict[str, float] = {}
    for run in client.search_runs([experiment.experiment_id], order_by=["start_time DESC"]):
        instrument = run.data.params.get("instrument")
        granularity = run.data.params.get("granularity")
        pair_label = f"{instrument}_{granularity}"

        if pair_label in pair_p_values:
            continue  # already have this pair's most recent run; skip older ones

        artifact_path = None
        for artifact in client.list_artifacts(run.info.run_id):
            if artifact.path.endswith("_predictions.npz"):
                artifact_path = artifact.path
                break
        if artifact_path is None:
            continue

        local_path = client.download_artifacts(run.info.run_id, artifact_path)
        data = np.load(local_path)
        lstm_correct = data["lstm_correct"]
        baseline_correct = data[f"{baseline}_correct"]

        if baseline == "persistence":
            # persistence_correct is one row shorter (see persistence_baseline) —
            # align by dropping the LSTM's corresponding first-row prediction.
            lstm_correct = lstm_correct[1:]

        pair_p_values[pair_label] = mcnemar_p_value(lstm_correct, baseline_correct)

    return benjamini_hochberg_report(pair_p_values, alpha=alpha)


def _print_report(report: dict[str, dict], total_pairs_expected: int | None = None) -> None:
    n_significant = sum(1 for r in report.values() if r["significant_after_correction"])
    if total_pairs_expected is not None and len(report) < total_pairs_expected:
        print(
            f"{len(report)} of {total_pairs_expected} expected pairs have data so far "
            f"({n_significant} significant after BH-FDR correction on what's available):\n"
        )
        print(
            "NOTE: as more pairs get trained, this correction will re-tighten across "
            "all of them -- a pair marked significant today can stop being significant "
            "once more pairs are added, purely because there are more tests to correct "
            "for, not because that pair's own result changed. Re-run this after each new "
            "pair rather than treating an early verdict as final.\n"
        )
    else:
        print(f"{len(report)} pairs tested, {n_significant} significant after BH-FDR correction:\n")
    for pair, result in sorted(report.items(), key=lambda kv: kv[1]["p_adjusted"]):
        flag = "SIGNIFICANT" if result["significant_after_correction"] else "not significant"
        print(f"  {pair:20s} p={result['p_value']:.4f}  p_adjusted={result['p_adjusted']:.4f}  {flag}")


def main() -> None:
    import argparse

    from forex_ml.config import load_params

    parser = argparse.ArgumentParser(
        description="Report which (instrument, granularity) pairs beat their baseline, BH-FDR corrected."
    )
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment", default="forex-lstm")
    parser.add_argument("--baseline", default="majority", choices=["majority", "persistence"])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--params", default=None, help="Path to params.yaml, to report progress against the full pair set")
    args = parser.parse_args()

    report = report_across_pairs(args.tracking_uri, args.experiment, args.baseline, args.alpha)

    total_pairs_expected = None
    try:
        params = load_params(args.params) if args.params else load_params()
        total_pairs_expected = len(params.feature.instruments) * len(params.feature.granularities)
    except FileNotFoundError:
        pass

    _print_report(report, total_pairs_expected)


if __name__ == "__main__":
    main()
