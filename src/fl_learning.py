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
from model.model_evaluate import model_evaluate, paired_bootstrap_auc_diff
from constants import LOCAL_VARS_VA, LOCAL_VARS_BMC, TARGET_COL, SHARED_VARS, CENTRAL_MODEL_BASE, CENTRAL_MODEL_BAK, NODE_MODEL_BASE,\
    NODE_MODEL_LOCAL_BASE, NODE_MODEL_LOCAL_SHARED_BASE, NODE_MODEL_LOCAL_FULL_BASE, NODE_MODEL_LOCAL_SHARED_OTHER_BAK
from dataset.get_dataset import DataFrameDataset
from utils.helpers import get_shared_model_state_dict, get_model_path, get_updated_model_state_dict, set_seed, save_run_config


class FederatedLearning:
    def __init__(self, round_index, data_source, other_data_source, data_dir, feature_cols, target_col, local_cols,
                 device='cpu', strategy='shared_only',
                 local_reg=None, global_reg=0.001, epochs=10, aggregation=False, log_dir='logs',
                 agg_weighting='sample_size', seed=42):
        super().__init__()
        # breakpoint()
        self.round_index = round_index
        self.data_source = data_source
        self.data_dir = data_dir
        self.other_data_source = other_data_source
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.local_cols = local_cols
        self.device = device
        self.strategy = strategy
        self.local_reg = local_reg
        self.global_reg = global_reg
        self.epochs = epochs
        self.aggregation = aggregation
        self.log_dir = log_dir
        self.agg_weighting = agg_weighting
        self.seed = seed

    def get_round_index(self):
        return self.rond_index

    def set_round_index(self, round_index):
        self.rond_index = round_index

    def create_datasets(self):
        """
        Load the train/test datasets for this site. There is no validation split.
        Returns:
            tuple: (train_dataset, test_dataloader)
        """
        train_data_path = os.path.join(self.data_dir, "train.csv")
        train_data = pd.read_csv(train_data_path)
        train_dataset = DataFrameDataset(train_data, self.feature_cols, self.target_col, self.local_cols)
        test_data_path = os.path.join(self.data_dir, "test.csv")
        test_data = pd.read_csv(test_data_path)
        test_dataset = DataFrameDataset(test_data, self.feature_cols, self.target_col, self.local_cols)
        test_dataloader = get_dataloader(test_dataset, one_batch=True, shuffle=False)
        return train_dataset, test_dataloader

    def train_model(self, next_round=False):
        train_dataset, _ = self.create_datasets()
        model = LogisticRegressionModel(input_dim=len(self.feature_cols))
        if next_round:
            # Load central model
            central_model_path = os.path.join(CENTRAL_MODEL_BASE, os.listdir(CENTRAL_MODEL_BASE)[0])
            central_model_state_dict = torch.load(central_model_path)
            assert central_model_state_dict['linear.weight'].shape[1] == len(self.feature_cols), \
                "Model input dimension does not match feature columns."
            model.load_state_dict(central_model_state_dict)
            model.to(self.device)
            print(f"Loaded updated model for next round training for data source {self.data_source}.")

        model_state_dict, sample_size = train(
            train_dataset,
            model,
            strategy=self.strategy,
            local_reg=self.local_reg,
            global_reg=self.global_reg,
            epochs=self.epochs)
        return {"model_state_dict": model_state_dict, "sample_size": sample_size}


    def eval_model(self, model_type='local', data_source=None):
        # breakpoint()
        model = LogisticRegressionModel(input_dim=len(self.feature_cols))
        train_dataset, test_dataloader = self.create_datasets()
        if model_type == 'local':
            all_model_files = [f for f in os.listdir(NODE_MODEL_LOCAL_SHARED_BASE) if self.data_source.lower() in f]
            if len(all_model_files) == 0:
                raise FileNotFoundError(f"No local model found for data source {self.data_source} in {NODE_MODEL_LOCAL_SHARED_BASE}.")
            elif len(all_model_files) > 1:
                warnings.warn(f"Multiple local models found for data source {self.data_source}. Using the latest one.")
                # Sort the already-filtered (this-site-only) list rather than indexing the
                # raw, unfiltered directory listing, which can otherwise return another
                # site's file. Filenames embed a sortable YYYYMMDD_HHMMSS timestamp.
                model_path = os.path.join(NODE_MODEL_LOCAL_SHARED_BASE, sorted(all_model_files)[-1])
            else:
                model_path = os.path.join(NODE_MODEL_LOCAL_SHARED_BASE, all_model_files[0])
        elif model_type == 'central':
            model_path = os.path.join(CENTRAL_MODEL_BASE, os.listdir(CENTRAL_MODEL_BASE)[-1])
        elif model_type == 'remote':
            model_files = os.listdir(NODE_MODEL_BASE)
            model_file = [f for f in model_files if data_source.lower() in f]
            assert len(model_file) == 1, f"Expected one remote model for data source {data_source}, found {len(model_file)}."
            model_path = os.path.join(NODE_MODEL_BASE, model_file[0])
        print(f"Model: {model_path} will be used for evaluation")
        model_state_dict = torch.load(model_path)
        if model_type == 'central':
            model.load_state_dict(model_state_dict)
        else:
            model.load_state_dict(model_state_dict['model_state_dict'])
        model.to(self.device)

        # Compute the model's predictions on its own training set ONLY to derive a
        # Youden's-J-optimal threshold (never from test data -- see
        # model_evaluate.youden_threshold docstring for why).
        train_dataloader = get_dataloader(train_dataset, one_batch=True, shuffle=False)
        train_results = model_evaluate(train_dataloader, model, bootstrap=False)

        results = model_evaluate(
            test_dataloader, model,
            train_probs=train_results['probs'], train_targets=train_results['targets'])
        if 'bootstrap' in results:
            print(f"Model accuracy on test set: {results['accuracy']:.4f} "
                  f"(Bootstrap: {results['bootstrap']['accuracy']['mean']:.4f} "
                  f"[{results['bootstrap']['accuracy']['ci_lower']:.4f}, "
                  f"{results['bootstrap']['accuracy']['ci_upper']:.4f}])")
            print(f"Model ROC AUC on test set: {results['roc_auc']:.4f} "
                  f"(Bootstrap: {results['bootstrap']['roc_auc']['mean']:.4f} "
                  f"[{results['bootstrap']['roc_auc']['ci_lower']:.4f}, "
                  f"{results['bootstrap']['roc_auc']['ci_upper']:.4f}])")
        else:
            print(f"Model accuracy on test set: {results['accuracy']:.4f}")
            print(f"Model ROC AUC on test set: {results['roc_auc']:.4f}")
        print(f"Brier score: {results['brier_score']:.4f}, PR-AUC: {results['pr_auc']:.4f}, "
              f"Calibration slope/intercept: {results['calibration_slope']:.4f}/{results['calibration_intercept']:.4f}")
        return results

    def model_aggregation(self):
        models = []
        # Load models from node model paths
        model_paths = [os.path.join(NODE_MODEL_BASE, f) for f in os.listdir(NODE_MODEL_BASE) if f.endswith('.pt')]
        for model_path in model_paths:
            model = torch.load(model_path)
            models.append(model)

        aggregated_params = model_aggregate(models, self.device, weighting=self.agg_weighting)
        return aggregated_params

    def save_model(self, model, model_type='local'):
        if model_type == 'local':
            datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = get_model_path(
                NODE_MODEL_LOCAL_SHARED_BASE, self.data_source, datetime_str, self.round_index)
            torch.save({
                'model_state_dict': model['model_state_dict'],
                'sample_size': model['sample_size']
            }, model_path)
        elif model_type == 'remote':
            datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path_remote = get_model_path(
                NODE_MODEL_BASE, self.data_source, datetime_str, self.round_index)
            node_name = self.data_source.lower()
            # remove existing models in node model base
            for f in os.listdir(NODE_MODEL_BASE):
                if node_name in f:
                    os.remove(os.path.join(NODE_MODEL_BASE, f))
            # Save checkpoint to remote path
            torch.save({
                'model_state_dict': model['model_state_dict'],
                'sample_size': model['sample_size']
            }, model_path_remote)
            # Save model from other data source in for backup of model from other datasource
            other_node_name = self.other_data_source.lower()
            for f in os.listdir(NODE_MODEL_BASE):
                if other_node_name in f:
                    os.makedirs(NODE_MODEL_LOCAL_SHARED_OTHER_BAK, exist_ok=True)
                    shutil.copy(os.path.join(NODE_MODEL_BASE, f),
                                os.path.join(NODE_MODEL_LOCAL_SHARED_OTHER_BAK, f))
        elif model_type == 'central':
            datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            central_model_path = os.path.join(
                CENTRAL_MODEL_BASE, f"central_model_{datetime_str}_round_{self.round_index}.pt")
            if not os.path.exists(CENTRAL_MODEL_BASE):
                os.makedirs(CENTRAL_MODEL_BASE)
            # clear existing models in central model base
            for f in os.listdir(CENTRAL_MODEL_BASE):
                os.makedirs(CENTRAL_MODEL_BAK, exist_ok=True)
                shutil.move(os.path.join(CENTRAL_MODEL_BASE, f),
                            os.path.join(CENTRAL_MODEL_BAK, f))
            torch.save(model, central_model_path)

    def save_logs(self, results, prefix="before_aggregation"):
        log_path = os.path.join(
            self.log_dir,
            f"{self.strategy}_{self.round_index}_{prefix}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
        with open(log_path, 'wb') as f:
            pkl.dump(results, f)

    def save_run_manifest(self, extra=None):
        """
        Save a run-manifest JSON capturing hyperparameters, seed, and package versions
        for this round's training invocation -- one file per round, alongside that
        round's logs, for 1:1 traceability.
        """
        config = {
            'round_index': self.round_index,
            'data_source': self.data_source,
            'strategy': self.strategy,
            'learning_rate': 1e-2,
            'batch_size': 64,
            'epochs': self.epochs,
            'local_reg': self.local_reg,
            'global_reg': self.global_reg,
            'agg_weighting': self.agg_weighting,
            'seed': self.seed,
        }
        if extra:
            config.update(extra)
        filename = f"run_config_{self.data_source}_{self.round_index}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        return save_run_config(self.log_dir, filename, config)

    def run(self, step="train-eval"):
        # Step 1: Train local model
        # breakpoint()
        next_round = False
        if self.round_index > 1:
            next_round = True
        if step == "train-eval":
            self.save_run_manifest()
            print(f"Starting training for round {self.round_index} for data source {self.data_source}...")
            model = self.train_model(next_round=next_round)
            self.save_model(model, model_type='local')
            self.save_model(model, model_type='remote')
            print(f"Training completed for round {self.round_index} for data source {self.data_source}.")
            # Step 2: Evaluate local model
            print(f"Starting evaluation for round {self.round_index} for data source {self.data_source}...")
            results = self.eval_model(model_type='local')
            print(f"Evaluation completed for round {self.round_index} for data source {self.data_source}.")
            self.save_logs(results, prefix="before_aggregation")

        if step == "aggregation":
            # Step 3: Model aggregation
            print(f"Starting model aggregation for round {self.round_index} (weighting={self.agg_weighting})...")
            central_model = self.model_aggregation()
            self.save_model(central_model, model_type='central')
            print(f"Model aggregation completed for round {self.round_index}.")
        if step == "eval-after-aggregation":
            # Step 4: model evaluation after aggregation
            print(f"Starting evaluation after aggregation for round {self.round_index} for data source {self.data_source}...")
            results_central = self.eval_model(model_type='central')
            self.save_logs(results_central, prefix="after_aggregation")
            print(f"Evaluation after aggregation completed for round {self.round_index} for data source {self.data_source}.")
            print(f"Starting re-evaluation of local model for round {self.round_index} for data source {self.data_source}...")
            results_local = self.eval_model(model_type='local')
            self.save_logs(results_local, prefix="after_aggregation_local_reeval")
            print(f"Starting cross evaluation after aggregation for round {self.round_index} for data source {self.data_source}...")
            results_cross = self.eval_model(model_type='remote', data_source=self.other_data_source)
            self.save_logs(results_cross, prefix="after_aggregation_cross")
            print(f"Cross Evaluation after aggregation completed for round {self.round_index} for data source {self.data_source}.")

            # Formal statistical comparison between model AUCs. All three models above
            # were just evaluated on this SAME site's test set, so a paired bootstrap
            # comparison is valid here.
            comparisons = {
                'federated_vs_local': paired_bootstrap_auc_diff(
                    results_central['probs'], results_local['probs'], results_central['targets']),
                'federated_vs_cross_site': paired_bootstrap_auc_diff(
                    results_central['probs'], results_cross['probs'], results_central['targets']),
            }
            fvl = comparisons['federated_vs_local']
            fvc = comparisons['federated_vs_cross_site']
            print(f"Federated vs local AUC diff: {fvl['observed_diff']:.4f} "
                  f"[{fvl['ci_lower']:.4f}, {fvl['ci_upper']:.4f}] (significant={fvl['significant']})")
            print(f"Federated vs cross-site AUC diff: {fvc['observed_diff']:.4f} "
                  f"[{fvc['ci_lower']:.4f}, {fvc['ci_upper']:.4f}] (significant={fvc['significant']})")
            self.save_logs(comparisons, prefix="after_aggregation_auc_comparisons")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the logistic regression model.")
    parser.add_argument("--env_path", type=str, default=".env")
    parser.add_argument("--round_index", type=int, default=1)
    parser.add_argument("--step", type=str, default="train-eval",
                        help="Step to run: 'train-eval', 'aggregation', 'eval-after-aggregation'. Default is 'train-eval'.")
    parser.add_argument("--data_source", type=str, required=True,
                        help="Specify data source used"),
    parser.add_argument("--other_data_source", type=str, required=True,
                        help="Specify the other site data source"),
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument('--strategy', type=str, default='shared_only',
                        help="Model training strategy. Options: 'shared_only', 'full'. Default is 'shared_only'.")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--local_reg", type=float,
                        help="Add L2 regularization to local variables.")
    group.add_argument("--global_reg", type=float,
                        help="Add L2 regularization to all variables.")
    parser.set_defaults(global_reg=0.001)
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
    # ---------------------------------------FL learning ----------------------------
    # ###############################################################################
    fl_round = FederatedLearning(
        round_index=args.round_index,
        data_source=args.data_source,
        data_dir=data_dir,
        other_data_source=args.other_data_source,
        feature_cols=feature_cols,
        target_col=target_col,
        local_cols=local_cols,
        device=args.device,
        strategy=args.strategy,
        local_reg=args.local_reg,
        global_reg=args.global_reg,
        epochs=args.epochs,
        aggregation=False,
        log_dir=args.log_dir,
        agg_weighting=args.agg_weighting,
        seed=args.seed
    )
    fl_round.run(args.step)
