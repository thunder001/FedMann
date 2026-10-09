import torch
import statsmodels.api as sm
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, roc_auc_score,
    roc_curve,
)
import numpy as np


def bootstrap_evaluate(probs, targets, n_bootstrap=2000, confidence_level=0.95):
    """
    Perform bootstrap resampling to estimate confidence intervals for metrics.
    Args:
        probs: Array of predicted probabilities.
        targets: Array of true labels.
        n_bootstrap: Number of bootstrap replicates.
        confidence_level: Confidence level for intervals (default 0.95).
    Returns:
        dict: Bootstrap statistics including mean, std, and confidence intervals.
    """
    probs = np.array(probs)
    targets = np.array(targets)
    n_samples = len(targets)

    bootstrap_accuracy = []
    bootstrap_roc_auc = []

    np.random.seed(42)  # For reproducibility

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_probs = probs[indices]
        boot_targets = targets[indices]
        boot_preds = [1 if p >= 0.5 else 0 for p in boot_probs]

        # Calculate metrics on bootstrap sample
        try:
            boot_accuracy = accuracy_score(boot_targets, boot_preds)
            boot_roc_auc = roc_auc_score(boot_targets, boot_probs)
            bootstrap_accuracy.append(boot_accuracy)
            bootstrap_roc_auc.append(boot_roc_auc)
        except ValueError:
            # Skip if bootstrap sample has only one class
            continue

    # Calculate statistics
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    bootstrap_stats = {
        'accuracy': {
            'mean': np.mean(bootstrap_accuracy),
            'std': np.std(bootstrap_accuracy),
            'ci_lower': np.percentile(bootstrap_accuracy, lower_percentile),
            'ci_upper': np.percentile(bootstrap_accuracy, upper_percentile),
            'bootstrap_values': bootstrap_accuracy
        },
        'roc_auc': {
            'mean': np.mean(bootstrap_roc_auc),
            'std': np.std(bootstrap_roc_auc),
            'ci_lower': np.percentile(bootstrap_roc_auc, lower_percentile),
            'ci_upper': np.percentile(bootstrap_roc_auc, upper_percentile),
            'bootstrap_values': bootstrap_roc_auc
        },
        'n_bootstrap': n_bootstrap,
        'confidence_level': confidence_level
    }

    return bootstrap_stats


def youden_threshold(probs, targets):
    """
    Compute the Youden's J-optimal classification threshold: J = sensitivity +
    specificity - 1 (equivalently TPR - FPR), the point on the ROC curve farthest above
    the chance diagonal.

    Intended usage: call this on TRAINING-set predictions only, then reuse the returned
    threshold to compute sensitivity/specificity/PPV/NPV on the test set. Deriving the
    threshold from the test set itself would let test-set information influence how
    test-set performance is reported, so this must always be computed from training
    data, even though that makes it an in-sample (not independently validated) choice.
    Args:
        probs: Predicted probabilities.
        targets: True binary labels.
    Returns:
        float: Threshold maximizing Youden's J.
    """
    fprs, tprs, thresholds = roc_curve(targets, probs)
    j_scores = tprs - fprs
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx])


def threshold_metrics(probs, targets, threshold):
    """
    Compute sensitivity, specificity, PPV, and NPV at a given classification threshold.
    Args:
        probs: Predicted probabilities.
        targets: True binary labels.
        threshold (float): Classification threshold.
    Returns:
        dict: threshold used, sensitivity, specificity, ppv, npv, and the underlying
            confusion-matrix counts.
    """
    probs = np.asarray(probs, dtype=float)
    targets = np.asarray(targets, dtype=int)
    preds = (probs >= threshold).astype(int)

    tp = int(np.sum((preds == 1) & (targets == 1)))
    tn = int(np.sum((preds == 0) & (targets == 0)))
    fp = int(np.sum((preds == 1) & (targets == 0)))
    fn = int(np.sum((preds == 0) & (targets == 1)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    npv = tn / (tn + fn) if (tn + fn) > 0 else float('nan')

    return {
        'threshold': float(threshold),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }


def calibration_metrics(probs, targets, n_bins=10):
    """
    Compute Brier score, a calibration curve, and calibration intercept/slope.

    Calibration intercept/slope are estimated via Cox calibration: a logistic
    recalibration model fit as logit(y) ~ 1 + logit(p_hat), using the already-fitted
    model's own predicted probabilities as the sole predictor. An intercept of 0 and
    slope of 1 indicate perfect calibration.
    Args:
        probs: Predicted probabilities.
        targets: True binary labels.
        n_bins (int): Number of bins for the calibration curve.
    Returns:
        dict: brier_score, calibration_intercept, calibration_slope, and
            calibration_curve (mean predicted probability and observed fraction
            positive per bin).
    """
    probs = np.asarray(probs, dtype=float)
    targets = np.asarray(targets, dtype=float)

    brier = brier_score_loss(targets, probs)

    n_bins_used = min(n_bins, len(np.unique(probs)))
    n_bins_used = max(n_bins_used, 1)
    frac_pos, mean_pred = calibration_curve(targets, probs, n_bins=n_bins_used, strategy='quantile')

    eps = 1e-6
    clipped = np.clip(probs, eps, 1 - eps)
    logit_p = np.log(clipped / (1 - clipped))
    design = sm.add_constant(logit_p)
    try:
        cal_model = sm.Logit(targets, design).fit(disp=0)
        intercept, slope = float(cal_model.params[0]), float(cal_model.params[1])
    except Exception:
        intercept, slope = float('nan'), float('nan')

    return {
        'brier_score': float(brier),
        'calibration_intercept': intercept,
        'calibration_slope': slope,
        'calibration_curve': {
            'mean_predicted': mean_pred.tolist(),
            'fraction_positive': frac_pos.tolist(),
            'n_bins': n_bins_used,
        },
    }


def paired_bootstrap_auc_diff(probs_a, probs_b, targets, n_bootstrap=2000,
                               confidence_level=0.95, seed=42):
    """
    Compare two models' AUCs on the SAME test set via paired bootstrap resampling.

    For each of `n_bootstrap` replicates, the SAME resampled indices are used to score
    BOTH models, and the AUC difference (model_a - model_b) is recorded for that
    replicate -- this "paired" resampling correctly accounts for the two AUC estimates
    being correlated (same underlying patients), which comparing two separately computed
    bootstrap CIs for overlap does not. This makes fewer distributional assumptions than
    a DeLong's test, which matters for small test sets.
    Args:
        probs_a: Predicted probabilities from model A on the test set.
        probs_b: Predicted probabilities from model B on the SAME test set.
        targets: True labels for the (shared) test set.
        n_bootstrap (int): Number of paired bootstrap replicates.
        confidence_level (float): Confidence level for the difference CI.
        seed (int): Random seed for reproducibility.
    Returns:
        dict: observed difference (A - B), bootstrap mean/std of the difference, the CI
            bounds, and whether the CI excludes zero ('significant').
    """
    probs_a = np.asarray(probs_a, dtype=float)
    probs_b = np.asarray(probs_b, dtype=float)
    targets = np.asarray(targets, dtype=float)
    n = len(targets)

    observed_diff = float(roc_auc_score(targets, probs_a) - roc_auc_score(targets, probs_b))

    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_targets = targets[idx]
        if len(np.unique(boot_targets)) < 2:
            continue
        auc_a = roc_auc_score(boot_targets, probs_a[idx])
        auc_b = roc_auc_score(boot_targets, probs_b[idx])
        diffs.append(auc_a - auc_b)

    alpha = 1 - confidence_level
    ci_lower = float(np.percentile(diffs, (alpha / 2) * 100))
    ci_upper = float(np.percentile(diffs, (1 - alpha / 2) * 100))

    return {
        'observed_diff': observed_diff,
        'bootstrap_mean_diff': float(np.mean(diffs)),
        'bootstrap_std_diff': float(np.std(diffs)),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_bootstrap': len(diffs),
        'confidence_level': confidence_level,
        'significant': not (ci_lower <= 0 <= ci_upper),
    }


def model_evaluate(data_loader, model, bootstrap=True, n_bootstrap=2000,
                    train_probs=None, train_targets=None):
    """
    Evaluate the model on the given data loader, with optional bootstrap resampling and
    extended performance metrics.
    Args:
        data_loader (DataLoader): DataLoader for the dataset (should always be a
            held-out set, e.g. test -- never the set the model was trained on).
        model (nn.Module): The model to evaluate.
        bootstrap (bool): Whether to perform bootstrap resampling (default True).
        n_bootstrap (int): Number of bootstrap replicates (default 2000).
        train_probs, train_targets (optional): The model's predicted probabilities and
            true labels on its OWN TRAINING set. If provided, these are used (and only
            these -- never test data) to derive a Youden's-J-optimal threshold, which is
            then applied to the test-set probabilities passed to this function. If
            omitted, only the fixed 0.5 threshold is reported.
    Returns:
        dict: Evaluation metrics, including bootstrap statistics if enabled, calibration
            metrics, PR-AUC, and threshold-based metrics (sensitivity/specificity/PPV/NPV)
            at 0.5 and, if train_probs/train_targets were given, at the Youden threshold.
    """
    model.eval()
    # breakpoint()
    probs, targets = [], []
    data_iter = iter(data_loader)
    with torch.no_grad():
        for batch in data_iter:
            batch_X = batch['features']
            batch_y = batch['target'].view(-1, 1)
            outputs = torch.sigmoid(model(batch_X))
            probs.extend(outputs.flatten().cpu().numpy())
            targets.extend(batch_y.flatten().cpu().numpy())

    probs = np.array(probs)
    targets = np.array(targets)
    preds = [1 if p >= 0.5 else 0 for p in probs]

    # Calculate metrics on full dataset
    accuracy = accuracy_score(targets, preds)
    roc_auc = roc_auc_score(targets, probs)
    fprs, tprs, _ = roc_curve(targets, probs)

    logs = {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "fprs": fprs,
        "tprs": tprs,
        "probs": probs.tolist(),
        "preds": preds,
        "targets": targets.tolist(),
    }

    # Extended metrics: calibration/Brier score, PR-AUC, and threshold-based metrics at
    # both a fixed 0.5 threshold and (if training predictions are supplied) a
    # training-set-derived Youden's-J-optimal threshold.
    logs.update(calibration_metrics(probs, targets))
    logs['pr_auc'] = float(average_precision_score(targets, probs))
    logs['threshold_metrics'] = {'0.5': threshold_metrics(probs, targets, 0.5)}
    if train_probs is not None and train_targets is not None:
        y_threshold = youden_threshold(train_probs, train_targets)
        logs['threshold_metrics']['youden_train'] = threshold_metrics(probs, targets, y_threshold)

    # Add bootstrap statistics if requested
    if bootstrap:
        bootstrap_stats = bootstrap_evaluate(probs, targets, n_bootstrap=n_bootstrap)
        logs['bootstrap'] = bootstrap_stats

    return logs
