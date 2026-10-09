# FedMann - a federated learning framework for small model

## Introduction

FedMann provides a data level privacy to enable model training among multiple sites. Different sites should have as many as overlapped
variables, but unique local variables are also allowed. The core design is to utilize github as a central location to share models. Each site uploads its local model to github using git operation.

FedMann impliments two strategies for model training. The default is "shared only", meaning local models are trained using common variables among all sites. Alternatively, the second strategy is called "full", meaning that local models are trained using all variables, including unique local varibles, but penalized by those variables in loss function, while central model is still based on common variables.

## Code base structure
The core component of FedMann is model directory, which includes two subdirectories: **central model** for aggregated model, and **node_model** for 
shared local models from different sites. In addition, there are several derectories to hold private local models, including node_model_local for never shared models, node_model_local_full for to-be-shared models trainied using all variables, and node_model_local_shared for models trainied using common variables. Another important non-shared model folder is **central_model_bak** that backups all central models locally, which facilitates model tracking and downstream analysis.

The src directory includes all scripts from data preparation, model architecture to model training and evaluation. The **preferred way** to use this framework is via the `fl_learning.py` script that streamlines the federated learning process. See the [Federated Learning Pipeline](#federated-learning-pipeline) section for details.

## Usage
### 1. Clone the repo
```bash
git clone https://github.com/thunder001/FedMann.git
```
### 2. Create a virtual environment
```bash
conda env create -f environment.yml
conda activate fedmann
```
### 3. Create .env and modify constants.py
According to the format in .env_example, add items in your own .env, including raw data location and your train/dev/test location.
`constants.py` has several important info for data processing, including target variable, shared variables and local variables, please modify this file properly.

### 4. Prepare data
```bash
python src/data_prep/data_prep.py --data_source [va|bmc]
```
**Note:** Replace `[va|bmc]` with your data source. This determines which local variables are used:
- `va`: Uses VA-specific local variables (`ADINationalRank`, `Income`)
- `bmc`: Uses BMC-specific local variables (`InsuranceType`, `Housing`, `LanguageGroup`)

**Note:** Raw column names must be harmonized to the canonical names in `constants.py` (identical across sites) before running this script -- it is not done automatically. Categorical values are cleaned and validated against `constants.CANONICAL_CATEGORIES` (whitespace/casing normalization plus known synonyms in `constants.VALUE_SYNONYM_MAP`); an unrecognized value raises a clear error rather than silently becoming a stray column.

**Note:** The data split is a 2-way 60% training / 40% test split (`--test_size 0.4` by default; there is no validation split -- see [Federated learning pipeline](#federated-learning-pipeline) below). The split happens **before** imputation and encoding: missing categorical/continuous values are imputed via chained-equations multiple imputation (`statsmodels.imputation.mice.MICEData`), fit on the training split only and applied to the test split using training-derived reference values, so no test-set information ever informs how a missing value gets filled. One-hot encoding then uses a fixed schema from `constants.CANONICAL_CATEGORIES`, so train and test always produce identical columns.

## Federated learning pipeline

To simulate the federated learning in real setting, the following steps should be followed and prefered.

### 1. Model training and evaluation
This step should be done at **each site**. When successful, each site needs to make a pull request to github and the
leading site will approve and merge it to master. The branch name for each site is recommended to use `round_<index>_<site>`

**At VA site:**
```bash
python src/fl_learning.py --round_index 1 --data_source va --other_data_source bmc --step train-eval
```

**At BMC site:**
```bash
python src/fl_learning.py --round_index 1 --data_source bmc --other_data_source va --step train-eval
```

This step:
- Trains a local model on the site's training data
- Evaluates the local model on the site's test set (saved as `before_aggregation` log)
- Saves the model locally and uploads to remote storage

### 2. Model aggregation
This step should be done only in assigned **leading site**. Once done, this site needs to update master branch to include aggregated model to share.
```bash
python src/fl_learning.py --round_index 1 --data_source va --other_data_source bmc --step aggregation
```

**Note:** The `--data_source` and `--other_data_source` arguments are required but not used during aggregation. You can use either site's values.

### 3. Model evaluation after aggregation
This step should be done at **each site** to evaluate the federated learning on master branch.

**At VA site:**
```bash
python src/fl_learning.py --round_index 1 --data_source va --other_data_source bmc --step eval-after-aggregation
```

**At BMC site:**
```bash
python src/fl_learning.py --round_index 1 --data_source bmc --other_data_source va --step eval-after-aggregation
```

This step:
- Evaluates the aggregated federated model on the site's test set (saved as `after_aggregation` log)
- Re-evaluates the site's own local model on the same test set (saved as `after_aggregation_local_reeval` log)
- Evaluates the other site's remote model on the site's test set (saved as `after_aggregation_cross` log)
- Runs a paired bootstrap-difference comparison between the federated model's AUC and each of the other two (saved as `after_aggregation_auc_comparisons` log), reporting a 95% CI on the AUC difference rather than relying on eyeballing whether two separate bootstrap CIs overlap

### Additional Options

**For Round 2 and beyond:**
Simply change the `--round_index` argument. The script automatically loads the central model from the previous round:
```bash
python src/fl_learning.py --round_index 2 --data_source va --other_data_source bmc --step train-eval
```

**With custom training parameters:**
```bash
# More epochs
python src/fl_learning.py --round_index 1 --data_source va --other_data_source bmc --step train-eval --epochs 20

# Custom regularization (for shared_only strategy)
python src/fl_learning.py --round_index 1 --data_source va --other_data_source bmc --step train-eval --global_reg 0.01

# Full strategy with local regularization
python src/fl_learning.py --round_index 1 --data_source va --other_data_source bmc --step train-eval --strategy full --local_reg 0.01

# Equal-weighting aggregation sensitivity analysis (1/K per site instead of the default
# sample-size-weighted nk/N) -- run only at the --step aggregation step
python src/fl_learning.py --round_index 1 --data_source va --other_data_source bmc --step aggregation --agg_weighting equal
```

After the first round is completed, above steps can be repeated until satisfied performance is obtained by only changing `--round_index` argument.

**Note:** Each site should have a `logs` folder to save model evaluation results before and after model aggregation. The logs include bootstrap confidence intervals (2000 replicates) for test evaluations, plus calibration (Brier score, calibration intercept/slope, calibration curve), PR-AUC, and sensitivity/specificity/PPV/NPV at both a fixed 0.5 threshold and a training-set-derived Youden's-J-optimal threshold. A `run_config_*.json` manifest is also saved to `logs/` for every `train-eval` invocation, recording the hyperparameters, random seed, and package versions used, for reproducibility.

## Training Strategies

FedMann supports two training strategies controlled by the `--strategy` argument:

### Strategy 1: Shared Variables Only (Default)
- **Trains on:** Only shared variables (common across all sites)
- **Saves to:** `model/node_model_local_shared/`
- **Use case:** Models that can be shared across different sites
- **Features used:** `SHARED_VARS` only

### Strategy 2: Full Variables (Shared + Local)
- **Trains on:** Both shared AND local variables
- **Saves to:** `model/node_model_local_full/`
- **Use case:** Models that leverage site-specific local variables
- **Features used:** `SHARED_VARS + LOCAL_VARS`

Both strategies can be used with the `fl_learning.py` script by adding `--strategy shared_only` or `--strategy full` to the commands above.

## Alternative: Using main.py Directly

For advanced users who need more control, you can use `src/main.py` directly. However, this requires manually coordinating the steps and is not recommended for standard federated learning workflows.

### Model Training
```bash
python src/main.py --mode model_training --data_source [va|bmc] --strategy shared_only --upload_model
```

### Model Aggregation
```bash
python src/main.py --mode model_aggregation
```

### Model Evaluation
```bash
python src/main.py --mode model_evaluation --data_source [va|bmc] --strategy shared_only
```

### Multiple Rounds
```bash
# Continue training from previous round (loads existing models)
python src/main.py --mode model_training --data_source bmc --strategy shared_only --next_round --upload_model
```

## Command Line Arguments Reference

### Federated Learning Script (`src/fl_learning.py`) - **Recommended**

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--round_index` | int | `1` | No | Specify the round to track |
| `--step` | str | `train-eval` | No | Step to run: `train-eval`, `aggregation`, `eval-after-aggregation` |
| `--data_source` | str | - | **Yes** | Data source for current site: `va` or `bmc` |
| `--other_data_source` | str | - | **Yes** | Data source for other site: `va` or `bmc` |
| `--strategy` | str | `shared_only` | No | Training strategy: `shared_only` or `full` |
| `--epochs` | int | `10` | No | Number of training epochs |
| `--local_reg` | float | `None` | No | L2 regularization for local variables (full strategy only) |
| `--global_reg` | float | `0.001` | No | L2 regularization for all variables (shared_only strategy) |
| `--device` | str | `cpu` | No | Device for training: `cpu` or `cuda` |
| `--env_path` | str | `.env` | No | Path to environment file |
| `--log_dir` | str | `logs` | No | Folder to save model evaluation results |
| `--agg_weighting` | str | `sample_size` | No | Aggregation weighting: `sample_size` (nk/N) or `equal` (1/K), used at the `aggregation` step |
| `--seed` | int | `42` | No | Random seed (weight init, batch shuffling, bootstrap) |

### Main Script (`src/main.py`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | `model aggregation` | Operation mode: `model_training`, `model_aggregation`, `model_evaluation` |
| `--round_index` | int | `1` | Specify the round to track |
| `--data_source` | str | `va` | Data source: `va` or `bmc` |
| `--strategy` | str | `shared_only` | Training strategy: `shared_only` or `full` |
| `--upload_model` | flag | `False` | Upload trained model to remote repository |
| `--next_round` | flag | `False` | Continue training from existing models |
| `--epochs` | int | `10` | Number of training epochs |
| `--local_reg` | float | `None` | L2 regularization for local variables (full strategy only) |
| `--global_reg` | float | `0.001` | L2 regularization for all variables (shared_only strategy) |
| `--device` | str | `cpu` | Device for training: `cpu` or `cuda` |
| `--env_path` | str | `.env` | Path to environment file |
| `--log_dir` | str | `logs` | Folder to save model evaluation results |
| `--agg_weighting` | str | `sample_size` | Aggregation weighting: `sample_size` (nk/N) or `equal` (1/K), used with `--mode model_aggregation` |
| `--seed` | int | `42` | Random seed (weight init, batch shuffling, bootstrap) |

### Data Preparation Script (`src/data_prep/data_prep.py`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_source` | str | `va` | Data source: `va` or `bmc` |
| `--env_path` | str | `.env` | Path to environment file |
| `--test_size` | float | `0.4` | Test-set fraction of the 2-way train/test split (default: 60/40 train/test) |
| `--seed` | int | `42` | Random seed for the split and imputation |

**Note:** The data split is a 2-way 60% training / 40% test split. There is no validation split -- see the [Federated learning pipeline](#federated-learning-pipeline) section for why.

## Model and Log Directory Structure

```
model/
├── central_model/           # Aggregated models from all sites (shared)
├── central_model_bak/       # Local backup for aggregated models (not shared)
├── node_model/              # Shared local models (shared)
├── node_model_local/        # Private local models (not shared)
├── node_model_local_shared/ # Models trained on shared variables only (not shared)
└── node_model_local_full/   # Models trained on all variables (not shared)
logs/                        # Model evaluation results (not shared)
```
