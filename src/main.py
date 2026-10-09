import argparse
import torch # type: ignore
import datetime
import pandas as pd # type: ignore
import os
import shutil
import json
import pickle as pkl
import pdb
from dotenv import load_dotenv # type: ignore
import json
import warnings
from os.path import dirname, realpath
import sys
sys.path.append(dirname(realpath(__file__)))

from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import roc_auc_score as auroc_score, accuracy_score # type: ignore

from model.train import train
from model.model_aggregate import model_aggregate
from model.model import LogisticRegressionModel
from dataset.get_dataset import get_dataloader
from model.model_evaluate import model_evaluate
from constants import LOCAL_VARS_VA, LOCAL_VARS_BMC, TARGET_COL, SHARED_VARS, CENTRAL_MODEL_BASE, NODE_MODEL_BASE,\
    NODE_MODEL_LOCAL_BASE, NODE_MODEL_LOCAL_SHARED_BASE, NODE_MODEL_LOCAL_FULL_BASE, CENTRAL_MODEL_BAK
from dataset.get_dataset import DataFrameDataset
from utils.helpers import get_shared_model_state_dict, get_model_path, get_updated_model_state_dict, \
    logisticregressionsklearn, set_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the logistic regression model.")
    parser.add_argument("--env_path", type=str, default=".env")
    parser.add_argument("--round_index", type=int, default=1)
    parser.add_argument("--data_source", type=str, default="va"),
    parser.add_argument("--next_round", action='store_true', default=False,
                        help="Whether to run the next round of training.")
    parser.add_argument("--upload_model", action='store_true', default=False,
                        help="Whether to upload the model to the server.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mode", type=str, default="model aggregation", 
                        help="Specify type of model to run." \
                        "Options: 'model_training', 'model_aggregation', 'sklearn_comparison'. Default is 'model training'.")
    parser.add_argument('--strategy', type=str, default='shared_only',
                        help="Model training strategy. Options: 'shared_only', 'full'. Default is 'shared_only'.")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--local_reg", type=float,
                        help="Add L2 regularization to local variables.")
    group.add_argument("--global_reg", type=float,
                        help="Add L2 regularization to all variables.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs for training. Default is 10")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Path to node model directory.")
    parser.add_argument("--agg_weighting", type=str, default="sample_size", choices=["sample_size", "equal"],
                        help="Aggregation weighting scheme: 'sample_size' (default, FedAvg-style nk/N) "
                             "or 'equal' (1/K per site).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for training (weight init, batch shuffling, bootstrap).")

    # ###############################################################################
    # ------------------------------------ Setup-----------------------------------
    # ###############################################################################
    args = parser.parse_args()
    set_seed(args.seed)
    # check args
    if args.strategy not in ['shared_only', 'full']:
        raise ValueError("Invalid strategy specified. Choose 'shared_only' or 'full'.")
    if args.strategy == 'shared_only' and args.local_reg is not None:
        raise ValueError("Local regularization is not applicable for 'shared_only' strategy.")
    if args.strategy == 'full' and args.global_reg is not None:
        raise ValueError("Global regularization is not applicable for 'full' strategy.")
    # global_reg's 0.001 default only applies to the shared_only strategy (its previous
    # unconditional default, via parser.set_defaults(), made the 'full' strategy
    # unusable from the CLI: global_reg was never actually None, so the check above
    # always raised, even for the exact invocation documented in the README).
    if args.strategy == 'shared_only' and args.global_reg is None:
        args.global_reg = 0.001
    # The sklearn L-BFGS comparison model isn't strategy-aware (it applies one L2
    # strength across all feature_cols, unlike the PyTorch model's separate
    # local_reg/global_reg), so resolve a single value for it here: whichever
    # regularization strength applies under the chosen strategy.
    sklearn_reg = args.global_reg if args.strategy == 'shared_only' else args.local_reg
    if sklearn_reg is None:
        sklearn_reg = 0.001
    # load environment variables
    if args.env_path:
        load_dotenv(args.env_path)
        data_dir = os.getenv("DATA_DIR")
        # # The following train_data_path is used to for demo only
        # data_dir1 = os.getenv("DATA_DIR1")
        # data_dir2 = os.getenv("DATA_DIR2")
    # get mata data
    meta_data_path = os.path.join(data_dir, 'meta.json')
    meta_data = json.load(open(meta_data_path, 'r'))
    target_col = meta_data['target_col']
    local_cols = meta_data['local_cols']
    feature_cols = meta_data['feature_cols']
    # model training stategy adjustments
    if args.strategy == 'shared_only':
        feature_cols = [col for col in feature_cols if col not in local_cols]
        local_cols = []
    
    # ###############################################################################
    # ------------------------------------model training-----------------------------
    # ###############################################################################
    if args.mode == "model_training":
        # breakpoint()
        print('------Loading data----------')
        train_data_path = os.path.join(data_dir, "train.csv")
        meta_data_path = os.path.join(data_dir, 'meta.json')
        train_data = pd.read_csv(train_data_path)
        input_dim = len(feature_cols)
        train_dataset = DataFrameDataset(train_data, feature_cols, target_col, local_cols)
        # Note: there is no validation split.

        print('------Initializing model----------')
        # Initialize model
        model = LogisticRegressionModel(input_dim=input_dim).float()
        if args.next_round:
            # breakpoint()
            if args.strategy == 'full':
                # Load updated model with central shared parameters and local local parameters
                updated_model_state_dict = get_updated_model_state_dict(
                    CENTRAL_MODEL_BASE, NODE_MODEL_LOCAL_FULL_BASE, args.data_source, local_cols, feature_cols)
                assert updated_model_state_dict['linear.weight'].shape[1] == input_dim, \
                    "Model input dimension does not match feature columns."
                model.load_state_dict(updated_model_state_dict)
    
            elif args.strategy == 'shared_only':
                # Load central model
                central_model_path = os.path.join(CENTRAL_MODEL_BASE, os.listdir(CENTRAL_MODEL_BASE)[0])
                central_model_state_dict = torch.load(central_model_path)
                assert central_model_state_dict['linear.weight'].shape[1] == input_dim, \
                    "Model input dimension does not match feature columns."
                model.load_state_dict(central_model_state_dict)
        model.to(args.device)
        print('------Model training started----------') 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Train the model from scratch
            model_state_dict, sample_size = train(
                train_dataset,
                model,
                strategy=args.strategy,
                local_reg=args.local_reg,
                global_reg=args.global_reg,
                epochs=args.epochs)
        print('------Model training completed----------')

        # Sklearn L-BFGS comparison on same data (train/test)
        print('------Sklearn logistic regression (L-BFGS) on same data----------')
        test_data_path = os.path.join(data_dir, 'test.csv')
        test_data = pd.read_csv(test_data_path) if os.path.exists(test_data_path) else None
        logisticregressionsklearn(
            train_data, feature_cols, target_col,
            test_data=test_data, global_reg=sklearn_reg, epochs=args.epochs
        )

        # construct model save path.
        # breakpoint()
        datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.strategy == 'shared_only':
            model_path = get_model_path(NODE_MODEL_LOCAL_SHARED_BASE, args.data_source, datetime_str, args.round_index)
        elif args.strategy == 'full':
            model_path = get_model_path(NODE_MODEL_LOCAL_FULL_BASE, args.data_source, datetime_str, args.round_index)

        # Save checkpoint to local path
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({
            'model_state_dict': model_state_dict,
            'sample_size': sample_size
        }, model_path)
        print(f'------Model saved to local: {model_path} ----------')
        
        if args.upload_model:
            print(f'------Upload model to remote  ----------') 
            if args.strategy == 'full':
            # get partial model with shared parameters only
                model_state_dict = get_shared_model_state_dict(model_state_dict, local_cols, feature_cols)
            model_path_remote = get_model_path(NODE_MODEL_BASE, args.data_source, datetime_str, args.round_index)
            node_name = args.data_source.lower()
            # remove existing models in node model base
            for f in os.listdir(NODE_MODEL_BASE):
                if node_name in f:
                 os.remove(os.path.join(NODE_MODEL_BASE, f))
            # Save checkpoint to remote path
            torch.save({
                'model_state_dict': model_state_dict,
                'sample_size': sample_size
            }, model_path_remote)
            print(f'------Model uploaded to remote: {model_path_remote} ----------') 

    # ###############################################################################
    # --------------------------------sklearn comparison only------------------------
    # ###############################################################################
    elif args.mode == "sklearn_comparison":
        # Run sklearn L-BFGS on saved data only (no PyTorch training). Use same DATA_DIR as FL.
        print('------Loading data (from DATA_DIR)----------')
        train_data_path = os.path.join(data_dir, "train.csv")
        test_data_path = os.path.join(data_dir, 'test.csv')
        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path) if os.path.exists(test_data_path) else None
        print('------Sklearn logistic regression (L-BFGS) on same data----------')
        logisticregressionsklearn(
            train_data, feature_cols, target_col,
            test_data=test_data, global_reg=sklearn_reg, epochs=args.epochs
        )

    # ###############################################################################
    # --------------------------------model aggregation-----------------------------
    # ###############################################################################
    elif args.mode == "model_aggregation":
        # breakpoint()
        models = []
        # Load models from node model paths
        model_paths = [os.path.join(NODE_MODEL_BASE, f) for f in os.listdir(NODE_MODEL_BASE) if f.endswith('.pt')]
        for model_path in model_paths:
            model = torch.load(model_path)
            models.append(model)

        device = args.device
        aggregated_params = model_aggregate(models, device, weighting=args.agg_weighting)
        # Save aggregated model parameters
        datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        central_model_path = os.path.join(
            CENTRAL_MODEL_BASE, f"central_model_{datetime_str}_{args.round_index}.pt")
        if not os.path.exists(CENTRAL_MODEL_BASE):
            os.makedirs(CENTRAL_MODEL_BASE)
        # clear existing models in central model base
        # Note: shutil.move() already removes the source file as part of the move; do
        # not add an os.remove() on the same path afterward, it no longer exists.
        for f in os.listdir(CENTRAL_MODEL_BASE):
            os.makedirs(CENTRAL_MODEL_BAK, exist_ok=True)
            shutil.move(os.path.join(CENTRAL_MODEL_BASE, f),
                        os.path.join(CENTRAL_MODEL_BAK, f))
        torch.save(aggregated_params, central_model_path)
        print(f'------Aggregated model uploaded to remote: {central_model_path} ----------')
    # ###############################################################################
    # --------------------------------model evaluation------------------------------
    # ###############################################################################
    elif args.mode == "model_evaluation":
        # breakpoint()
        train_data_path = os.path.join(data_dir, "train.csv")
        test_data_path = os.path.join(data_dir, "test.csv")
        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)
        train_dataset = DataFrameDataset(train_data, feature_cols, target_col, local_cols)
        test_dataset = DataFrameDataset(test_data, feature_cols, target_col, local_cols)
        train_dataloader = get_dataloader(train_dataset, one_batch=True, shuffle=False)
        test_dataloader = get_dataloader(test_dataset, one_batch=True, shuffle=False)
        # Load the recent local model
        if args.strategy == 'shared_only':
            model_path = os.path.join(NODE_MODEL_LOCAL_SHARED_BASE, os.listdir(NODE_MODEL_LOCAL_SHARED_BASE)[-1])
        elif args.strategy == 'full':
            model_path = os.path.join(NODE_MODEL_LOCAL_FULL_BASE, os.listdir(NODE_MODEL_LOCAL_FULL_BASE)[-1])
        model_state_dict = torch.load(model_path)
        input_dim = len(feature_cols)
        assert model_state_dict['model_state_dict']['linear.weight'].shape[1] == input_dim, \
            "Model input dimension does not match feature columns."
        model = LogisticRegressionModel(input_dim=input_dim)
        model.load_state_dict(model_state_dict['model_state_dict'])
        model.to(args.device)
        # Evaluate the model. Train-set predictions are computed first (never reported
        # themselves, bootstrap disabled since they're only an intermediate) so the
        # training-derived Youden threshold can be applied to the test set below --
        # deriving it from the test set itself would let test-set information influence
        # how test-set performance is reported.
        train_results = model_evaluate(train_dataloader, model, bootstrap=False)
        results = model_evaluate(
            test_dataloader, model,
            train_probs=train_results['probs'], train_targets=train_results['targets']
        )
        print(f"Model accuracy on test set: {results['accuracy']:.4f} "
              f"(Bootstrap: {results['bootstrap']['accuracy']['mean']:.4f} "
              f"[{results['bootstrap']['accuracy']['ci_lower']:.4f}, "
              f"{results['bootstrap']['accuracy']['ci_upper']:.4f}])")
        print(f"Model ROC AUC on test set: {results['roc_auc']:.4f} "
              f"(Bootstrap: {results['bootstrap']['roc_auc']['mean']:.4f} "
              f"[{results['bootstrap']['roc_auc']['ci_lower']:.4f}, "
              f"{results['bootstrap']['roc_auc']['ci_upper']:.4f}])")
        print(f"PR-AUC on test set: {results['pr_auc']:.4f}")
        print(f"Brier score: {results['brier_score']:.4f}, "
              f"Calibration intercept: {results['calibration_intercept']:.4f}, "
              f"Calibration slope: {results['calibration_slope']:.4f}")
        for thresh_name, tm in results['threshold_metrics'].items():
            print(f"Threshold={thresh_name} (value={tm['threshold']:.3f}): "
                  f"sensitivity={tm['sensitivity']:.3f}, specificity={tm['specificity']:.3f}, "
                  f"ppv={tm['ppv']:.3f}, npv={tm['npv']:.3f}")
        # Save evaluation logs
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        log_path = os.path.join(args.log_dir, f"evaluation_log_{args.strategy}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
        with open(log_path, 'wb') as f:
            pkl.dump(results, f)
        print(f'------Evaluation log saved to: {log_path} ----------')

    else:
        raise ValueError("Invalid mode specified. Choose 'model_training', 'model_aggregation', 'sklearn_comparison', or 'model_evaluation'.")