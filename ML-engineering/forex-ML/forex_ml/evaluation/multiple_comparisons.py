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
  3. Groups by (pair, model configuration), not by pair alone — see
     `_model_config_signature`. Architecture search (different layer counts/widths,
     activation functions, epochs, regularization, learning rate, ANY training
     parameter — on the same pair) is itself a form of model selection: picking
     whichever configuration wins and reporting only that one's p-value is exactly
     the kind of researcher-degrees-of-freedom problem this whole module exists to
     guard against. Treating every distinct configuration tried as its own hypothesis
     keeps the correction honest as architecture search gets more frequent.
     Retraining the SAME configuration as more data accumulates over time (the
     single-local-GPU, incremental-data workflow this was originally built for) still
     collapses to its most recent run — that's genuinely one hypothesis re-evaluated
     with an updated estimate, not a new one.
"""

from __future__ import annotations

import hashlib
import json

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


def config_signature_from_params(params: dict) -> str:
    """Hash of every logged param except `instrument`/`granularity` (already captured
    by the pair label) and `run_uid` (unique per run by construction — including it
    would make every single run its own "configuration" and defeat the point). Two
    runs count as the SAME configuration only if every other logged param matches
    exactly: architecture (`number_of_cells_per_rnn_layer`/
    `number_of_cells_per_dense_layer`), activation functions, epochs, batch size,
    regularization, dropout, learning rate — the full `TrainParams` set logged in
    `train_and_evaluate`, not just layer count/width. Whole-params equality, not a
    hand-picked subset, is the point: any change a human might make between training
    attempts on the same pair should count as a distinct hypothesis for the
    multiple-comparisons correction below, not just the ones we thought to name.

    Values are stringified before hashing: `run.data.params` (below) is always
    already-stringified (MLflow stores every param as a string), but
    `forex_ml.training.train` also calls this directly with the pre-logging dict of
    native Python types, to tag a newly registered model version with the same
    signature multiple_comparisons would later compute for its run — str()'ing both
    call sites' values the same way keeps the two hashes identical for the same
    configuration.

    `long_swap_cost_pct_per_night`/`short_swap_cost_pct_per_night` are ALSO
    excluded, despite being real inputs to the labeling math -- since
    forex_ml.data.swap_rates.resolve_swap_cost_pct_per_night started resolving
    them from a live InfluxDB snapshot (see split_flow.py), they're
    environmentally-resolved values that can drift between two otherwise-identical
    retrains of the same hyperparameters on different days, not a hyperparameter a
    human chose. Without this exclusion, two runs a human considers "the same
    configuration, retrained" would get different signatures purely because
    OANDA's rate ticked in between, silently fracturing the BH-FDR pool below and
    the model registry's "one canonical config per (pair, hyperparams)"
    assumption. `swap_cost_pct_per_night` (the old, pre-bidirectional single-value
    name) stays excluded permanently too, even though nothing logs it under that
    exact name anymore -- real historical MLflow runs were logged with it, and
    dropping it from the excluded set would silently change THEIR config
    signatures the next time this function runs against them.
    """
    excluded = {
        "instrument", "granularity", "run_uid",
        "swap_cost_pct_per_night",
        "long_swap_cost_pct_per_night", "short_swap_cost_pct_per_night",
    }
    items = sorted((k, str(v)) for k, v in params.items() if k not in excluded)
    payload = json.dumps(items, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _model_config_signature(run) -> str:
    return config_signature_from_params(run.data.params)


def _config_summary(run) -> str:
    """Short human-readable architecture snapshot for print output only. Grouping
    itself is decided by `_model_config_signature`, which hashes every logged param,
    not just the ones summarized here."""
    keys = [
        "number_of_cells_per_rnn_layer", "number_of_cells_per_dense_layer",
        "lstm_activation_function", "dense_activation_function", "epochs",
    ]
    parts = [f"{k}={run.data.params[k]}" for k in keys if k in run.data.params]
    return ", ".join(parts) if parts else "(no architecture params logged)"


def report_across_pairs(
    tracking_uri: str,
    experiment_name: str,
    baseline: str = "majority",
    alpha: float = 0.05,
) -> dict[str, dict]:
    """Pulls every run in `experiment_name`, downloads each run's predictions.npz
    artifact (saved by forex_ml.training.train.train_and_evaluate), runs McNemar's
    test per (pair, model configuration), and BH-corrects across all of them.

    Runs are grouped by (instrument, granularity, `_model_config_signature`), not by
    pair alone. If the SAME configuration has been trained more than once for a pair
    (expected on a single local GPU, where data is built up incrementally over time),
    only its most recent run is used — runs are pulled ordered by start_time
    descending and older duplicates for an already-seen (pair, configuration) are
    skipped. A DIFFERENT configuration trained on the same pair (architecture search)
    is treated as a separate hypothesis with its own entry in the report and its own
    slot in the correction, rather than silently overwriting or being overwritten by
    another configuration's result.
    """
    if baseline not in ("majority", "persistence"):
        raise ValueError(f"baseline must be 'majority' or 'persistence', got {baseline!r}")

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"No MLflow experiment named {experiment_name!r}")

    pair_p_values: dict[str, float] = {}
    metadata: dict[str, dict] = {}
    for run in client.search_runs([experiment.experiment_id], order_by=["start_time DESC"]):
        instrument = run.data.params.get("instrument")
        granularity = run.data.params.get("granularity")
        config_sig = _model_config_signature(run)
        key = f"{instrument}_{granularity}::{config_sig}"

        if key in pair_p_values:
            continue  # already have this (pair, configuration)'s most recent run

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

        pair_p_values[key] = mcnemar_p_value(lstm_correct, baseline_correct)
        metadata[key] = {
            "instrument": instrument,
            "granularity": granularity,
            "config_summary": _config_summary(run),
        }

    report = benjamini_hochberg_report(pair_p_values, alpha=alpha)
    for key, meta in metadata.items():
        report[key].update(meta)
    return report


def _print_report(report: dict[str, dict]) -> None:
    n_significant = sum(1 for r in report.values() if r["significant_after_correction"])
    distinct_pairs = {(r["instrument"], r["granularity"]) for r in report.values()}
    print(
        f"{len(report)} (pair, configuration) combination(s) tested across "
        f"{len(distinct_pairs)} distinct pair(s), {n_significant} significant after "
        f"BH-FDR correction:\n"
    )
    print(
        "NOTE: this correction re-tightens every time a new configuration or pair is "
        "added -- a result marked significant today can stop being significant once "
        "more are added, purely because there are more tests to correct for, not "
        "because that result's own p-value changed. Re-run this after each new "
        "training run rather than treating an early verdict as final.\n"
    )
    for _, result in sorted(report.items(), key=lambda kv: kv[1]["p_adjusted"]):
        flag = "SIGNIFICANT" if result["significant_after_correction"] else "not significant"
        label = f"{result['instrument']}_{result['granularity']}"
        print(
            f"  {label:15s} [{result['config_summary']}]  "
            f"p={result['p_value']:.4f}  p_adjusted={result['p_adjusted']:.4f}  {flag}"
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Report which (instrument, granularity, model configuration) combinations "
                     "beat their baseline, BH-FDR corrected."
    )
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment", default="forex-lstm")
    parser.add_argument("--baseline", default="majority", choices=["majority", "persistence"])
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    report = report_across_pairs(args.tracking_uri, args.experiment, args.baseline, args.alpha)
    _print_report(report)


if __name__ == "__main__":
    main()
