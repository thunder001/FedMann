import importlib.metadata
import json
import os
import random
import sys

import numpy as np
import torch

from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import roc_auc_score as auroc_score, accuracy_score # type: ignore


def set_seed(seed):
    """
    Seed every source of randomness used by the pipeline (Python's random module, numpy,
    and torch) so a run is reproducible end to end, including model weight
    initialization and mini-batch shuffling.
    Args:
        seed (int): Random seed to use everywhere.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_run_config(log_dir, filename, config):
    """
    Save a JSON run manifest capturing the hyperparameters, seed, and package versions used
    for a given training/aggregation invocation, so the run can be reproduced or audited
    later. One file per invocation (not a single running manifest), saved alongside that
    invocation's model checkpoint/logs for easy 1:1 traceability.
    Args:
        log_dir (str): Directory to save the manifest into (created if missing).
        filename (str): Manifest filename, e.g. "run_config_bmc_round1_20260101_120000.json".
        config (dict): Run-specific fields (round index, data source, strategy,
            hyperparameters, seed, aggregation weighting mode, etc.).
    Returns:
        str: Full path the manifest was written to.
    """
    os.makedirs(log_dir, exist_ok=True)

    def _pkg_version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    manifest = dict(config)
    manifest["package_versions"] = {
        "python": sys.version.split()[0],
        "torch": _pkg_version("torch"),
        "scikit-learn": _pkg_version("scikit-learn"),
        "pandas": _pkg_version("pandas"),
        "statsmodels": _pkg_version("statsmodels"),
        "numpy": _pkg_version("numpy"),
    }
    path = os.path.join(log_dir, filename)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path


def get_shared_model_state_dict(model_state_dict, local_cols, feature_cols):
    """
    Extract the partial model parameters corresponding to the local columns.
    Args:
        model_state_dict (dict): State dictionary of the model.
        local_cols (list): List of local column names.
        feature_cols (list): List of all feature column names in the data.
    Returns:
        dict: Partial model state dictionary containing only local parameters.
    """
    local_cols_idx = [feature_cols.index(col) for col in local_cols]
    shared_cols_idx = [i for i in range(len(feature_cols)) if i not in local_cols_idx]
    shared_state_dict = {}
    for k, v in model_state_dict.items():
        if 'weight' in k:
            shared_weights = v[0][shared_cols_idx].unsqueeze(0)
            shared_state_dict[k] = shared_weights
        elif 'bias' in k:
            shared_state_dict[k] = v
        else:
            # Handle other layers if necessary
            shared_state_dict[k] = v
    return shared_state_dict


def get_model_path(node_model_base, data_source, datetime_str, round_index):
    """
    Construct the remote path for saving the model based on the data source and current datetime.
    Args:
        node_model_base (str): Base directory for saving models.
        data_source (str): Data source identifier (e.g., 'VA' or 'BMC').
        datetime_str (str): Current datetime string for uniqueness.
    Returns:
        str: Constructed remote model path.
    """
    if data_source == 'va':
        model_remote_path = os.path.join(
            node_model_base, f"model_va_{datetime_str}_round_{round_index}.pt")
    elif data_source == 'bmc':
        model_remote_path = os.path.join(
            node_model_base, f"model_bmc_{datetime_str}_round_{round_index}.pt")
    else:
        raise ValueError("Invalid data source. Must be 'VA' or 'BMC'.")
    return model_remote_path


def get_updated_model_state_dict(central_model_base, node_model_local_base, data_source, local_cols, feature_cols):
    """
    Get the updated model state dictionary by combining the central model's shared parameters
    with the local model's local parameters.
    Args:
        central_model_base (str): Directory containing the central model.
        node_model_base (str): Directory containing the local model.
        data_source (str): Data source identifier (e.g., 'VA' or 'BMC').
        local_cols (list): List of local column names.
    Returns:
        dict: Updated model state dictionary.
    """
    # breakpoint()
    # Load central model
    central_model_path = os.path.join(central_model_base, os.listdir(central_model_base)[0])
    central_model_state_dict = torch.load(central_model_path)
    
    # Load last local model
    local_model_files = [f for f in os.listdir(node_model_local_base) if data_source.lower() in f]
    last_model_file = sorted(local_model_files)[-1]
    local_model_path = os.path.join(node_model_local_base, last_model_file)
    local_model_state_dict = torch.load(local_model_path)['model_state_dict']
    
    # Update local model parameters using central model's shared parameters
    
    updated_model_state_dict = local_model_state_dict.copy()
    local_cols_idx = [feature_cols.index(col) for col in local_cols]
    shared_cols_idx = [i for i in range(len(feature_cols)) if i not in local_cols_idx]
    for k, v in local_model_state_dict.items():
        if 'weight' in k:
            updated_weights = v.clone()
            updated_weights[0][shared_cols_idx] = central_model_state_dict[k][0]
            updated_model_state_dict[k] = updated_weights
        elif 'bias' in k:
            updated_model_state_dict[k] = v  # Update bias directly
        else:
            # Handle other layers if necessary
            updated_model_state_dict[k] = v
    return updated_model_state_dict


def logisticregressionsklearn(train_data, feature_cols, target_col, test_data=None,
                              global_reg=0.001, epochs=10):
    """
    Train sklearn LogisticRegression (L-BFGS) on the same data as the PyTorch model,
    and evaluate on the training set and optionally a test set.
    Args:
        train_data: Training DataFrame.
        feature_cols: List of feature column names.
        target_col: Name of target column.
        test_data: Optional test DataFrame; if provided, test set metrics are computed.
        global_reg: L2 regularization strength (C = 1/global_reg in sklearn).
        epochs: Max iterations for L-BFGS (max_iter).
    Returns:
        dict with keys 'train_accuracy', 'train_roc_auc', and if test_data given
        'test_accuracy', 'test_roc_auc'.
    """
    X_train = train_data[feature_cols].values
    y_train = train_data[target_col].values
    model_sklearn = LogisticRegression(
        penalty='l2',
        C=1 / global_reg,
        solver='lbfgs',
        max_iter=epochs,
        fit_intercept=True,
        random_state=42
    )
    model_sklearn.fit(X_train, y_train)

    preds_train = model_sklearn.predict_proba(X_train)[:, 1]
    train_auc = auroc_score(y_train, preds_train)
    train_accuracy = accuracy_score(y_train, preds_train >= 0.5)
    print(f"Training ROC AUC (sklearn): {train_auc:.4f}")
    print(f"Training Accuracy (sklearn): {train_accuracy:.4f}")

    results = {'train_accuracy': train_accuracy, 'train_roc_auc': train_auc}

    if test_data is not None:
        X_test = test_data[feature_cols].values
        y_test = test_data[target_col].values
        preds_test = model_sklearn.predict_proba(X_test)[:, 1]
        test_auc = auroc_score(y_test, preds_test)
        test_accuracy = accuracy_score(y_test, preds_test >= 0.5)
        print(f"Test ROC AUC (sklearn): {test_auc:.4f}")
        print(f"Test Accuracy (sklearn): {test_accuracy:.4f}")
        results['test_accuracy'] = test_accuracy
        results['test_roc_auc'] = test_auc

    print('------Model training completed - sklearn----------')
    return results
