# -*- coding: utf-8 -*-
"""
Prints a readable summary of every .pkl log file in logs/, from either main.py's
model_evaluation mode (evaluation_log_*.pkl) or fl_learning.py's round-based logs
(<strategy>_<round>_<before|after>_aggregation[...]_*.pkl). Run from the FedMann root:
    python src/review_logs.py
Or point it at a different folder:
    python src/review_logs.py --log_dir path/to/logs
"""
import argparse
import glob
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", default="logs")
args = parser.parse_args()


def print_eval_result(r):
    """A model_evaluate() result: accuracy/roc_auc plus extended metrics."""
    print(f"Accuracy: {r['accuracy']:.4f}  |  ROC AUC: {r['roc_auc']:.4f}")
    if "bootstrap" in r:
        b = r["bootstrap"]
        print(f"  Accuracy bootstrap: {b['accuracy']['mean']:.4f} "
              f"[{b['accuracy']['ci_lower']:.4f}, {b['accuracy']['ci_upper']:.4f}]")
        print(f"  ROC AUC bootstrap:  {b['roc_auc']['mean']:.4f} "
              f"[{b['roc_auc']['ci_lower']:.4f}, {b['roc_auc']['ci_upper']:.4f}] "
              f"(n={b['n_bootstrap']}, {int(b['confidence_level'] * 100)}% CI)")
    if "pr_auc" in r:
        print(f"PR-AUC: {r['pr_auc']:.4f}")
    if "brier_score" in r:
        print(f"Brier score: {r['brier_score']:.4f}  |  "
              f"Calibration intercept: {r['calibration_intercept']:.4f}  |  "
              f"Calibration slope: {r['calibration_slope']:.4f}")
    if "threshold_metrics" in r:
        for name, tm in r["threshold_metrics"].items():
            print(f"  Threshold={name} (value={tm['threshold']:.3f}): "
                  f"sensitivity={tm['sensitivity']:.3f}, specificity={tm['specificity']:.3f}, "
                  f"ppv={tm['ppv']:.3f}, npv={tm['npv']:.3f}, "
                  f"tp={tm['tp']} tn={tm['tn']} fp={tm['fp']} fn={tm['fn']}")
    if "calibration_curve" in r:
        cc = r["calibration_curve"]
        print(f"Calibration curve ({cc['n_bins']} bins):")
        for mp, fp in zip(cc["mean_predicted"], cc["fraction_positive"]):
            print(f"    predicted={mp:.3f}  observed={fp:.3f}")


def print_bootstrap_diff(name, d):
    """A single paired_bootstrap_auc_diff() result."""
    print(f"  {name}: observed diff = {d['observed_diff']:.4f}, "
          f"95% CI [{d['ci_lower']:.4f}, {d['ci_upper']:.4f}], "
          f"significant={d['significant']} (n={d['n_bootstrap']})")


def print_auc_comparisons(r):
    """The after_aggregation_auc_comparisons dict: {name: paired_bootstrap_auc_diff result}."""
    print("AUC comparisons (paired bootstrap difference):")
    for name, d in r.items():
        if isinstance(d, dict) and "observed_diff" in d:
            print_bootstrap_diff(name, d)
        else:
            print(f"  {name}: {d}")


paths = sorted(glob.glob(os.path.join(args.log_dir, "*.pkl")))
if not paths:
    print(f"No .pkl log files found in {args.log_dir}/")
else:
    for path in paths:
        with open(path, "rb") as f:
            r = pickle.load(f)
        print(f"\n{'=' * 70}\n{os.path.basename(path)}\n{'=' * 70}")
        if isinstance(r, dict) and "roc_auc" in r:
            print_eval_result(r)
        elif isinstance(r, dict) and any(
            isinstance(v, dict) and "observed_diff" in v for v in r.values()
        ):
            print_auc_comparisons(r)
        else:
            print("Unrecognized log structure -- raw contents:")
            print(r)
