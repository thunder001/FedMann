import argparse
import json
import os
import pickle
import sys
import warnings
from os.path import dirname, realpath

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression, LogisticRegression

sys.path.append(dirname(dirname(realpath(__file__))))

from constants import (
    SHARED_VARS, LOCAL_VARS_VA, LOCAL_VARS_BMC, TARGET_COL, CONTINUOUS_VARS,
    CANONICAL_CATEGORIES, VALUE_SYNONYM_MAP, DIAGNOSIS_YEAR_CENTER, DIAGNOSIS_YEAR_SCALE,
    DIAGNOSIS_ERA_CUTPOINT,
)
from utils.helpers import set_seed


def train_test_split(data, test_size=0.4, random_state=None):
    """
    Splits data into a 2-way train/test split.
    Args:
        data (pd.DataFrame): The input data.
        test_size (float): Proportion of the data to hold out as the test set.
        random_state (int, optional): Random seed for reproducibility.
    Returns:
        tuple: (train_data, test_data)
    """
    data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    n_train = int(len(data) * (1 - test_size))
    train_data = data.iloc[:n_train].reset_index(drop=True)
    test_data = data.iloc[n_train:].reset_index(drop=True)
    return train_data, test_data


def get_local_cols(feature_cols, local_vars):
    """
    Get the local columns from the data based on the provided local variable names.
    Args:
        feature_cols (list): List of all feature column names in the data.
        local_vars (list): List of local variable names.
    Returns:
        list: List of local column names present in the data.
    """
    local_cols = [col for col in feature_cols if any(var in col for var in local_vars)]
    return local_cols


def clean_categorical_column(series, canonical_categories, synonym_map=None, column_name=None):
    """
    Clean and validate a single categorical column: strip whitespace, normalize casing
    via a known-synonym lookup, and confirm every remaining non-null value is one of the
    pre-specified canonical categories.

    Genuinely missing values (NaN) are left untouched here -- they are not coerced into a
    string or a category -- so the imputation step downstream can impute them properly.
    Anything else unrecognized raises a clear, informative error naming the offending
    value(s) and their row count, rather than silently letting them flow into one-hot
    encoding as a stray column (which would silently desynchronize feature columns
    between splits/sites and corrupt the fixed encoding schema).
    Pure casing/whitespace variants of a canonical value (e.g. " f " for "F") are
    auto-normalized to the canonical spelling without needing a synonym-map entry --
    that's unambiguous formatting noise, not a semantic judgment call. Anything that
    doesn't match even case-insensitively (a genuine typo, abbreviation, or unrecognized
    label) still requires an explicit constants.VALUE_SYNONYM_MAP entry or a
    constants.CANONICAL_CATEGORIES addition; it is never guessed.
    Args:
        series (pd.Series): Raw categorical column, possibly containing NaN.
        canonical_categories (list): Full set of valid canonical values for this
            variable (see constants.CANONICAL_CATEGORIES). If empty/falsy, validation
            is skipped (used for variables whose canonical category list isn't
            confirmed yet).
        synonym_map (dict, optional): Known variant-value -> canonical-value mapping,
            keyed by the lowercased, whitespace-stripped variant. A mapped value of
            None means "treat this raw value as missing" (e.g. a literal "Unknown"
            string that should be imputed rather than kept as its own category),
            rather than a variant spelling of a real category.
        column_name (str, optional): Name used in the error message.
    Returns:
        pd.Series: Cleaned column (missing values still NaN; everything else canonical).
    """
    synonym_map = synonym_map or {}
    name = column_name or series.name
    canonical_lookup = {c.lower(): c for c in canonical_categories} if canonical_categories else {}

    is_null = series.isna()
    # Force object dtype up front: a column with mixed numeric-looking values and NaN can
    # otherwise be float64, which raises when we try to assign cleaned string values into it.
    cleaned = series.astype(object)

    if (~is_null).any():
        non_null_raw = cleaned[~is_null].astype(str).str.strip()
        # Common artifact of numeric columns that got upcast to float because of NaNs
        # elsewhere in the column (e.g. a category code "1" stored as "1.0").
        non_null_raw = non_null_raw.str.replace(r"^(\d+)\.0$", r"\1", regex=True)
        normalized_keys = non_null_raw.str.lower()

        def _resolve(raw_value, normalized_value):
            if normalized_value in synonym_map:
                return synonym_map[normalized_value]
            if normalized_value in canonical_lookup:
                return canonical_lookup[normalized_value]
            return raw_value

        mapped = [_resolve(raw, norm) for raw, norm in zip(non_null_raw, normalized_keys)]
        cleaned.loc[~is_null] = mapped
        # A synonym_map entry may map a raw value to None, meaning "this is really a
        # missing value, not a category" -- recompute the null mask so those rows are
        # treated as missing (and excluded from the canonical-category check below)
        # rather than compared against the canonical list as the literal value None.
        is_null = cleaned.isna()

    if canonical_categories:
        observed_non_null = cleaned[~is_null]
        unexpected = observed_non_null[~observed_non_null.isin(canonical_categories)]
        if len(unexpected) > 0:
            bad_values = unexpected.value_counts().to_dict()
            raise ValueError(
                f"Column '{name}' contains {len(unexpected)} value(s) not in its canonical "
                f"category list {canonical_categories}: {bad_values}. If these are a known "
                f"variant spelling, add an entry to constants.VALUE_SYNONYM_MAP['{name}']; if "
                f"this is a legitimate new category, add it to "
                f"constants.CANONICAL_CATEGORIES['{name}']. Do not let it pass through "
                f"silently -- it would otherwise become a stray one-hot column."
            )
    return cleaned


def clean_dataframe(data, categorical_vars):
    """
    Apply clean_categorical_column() to every categorical predictor column present in
    `data`, using the canonical category / synonym definitions in constants.py.
    """
    for var in categorical_vars:
        if var not in data.columns:
            continue
        canonical = CANONICAL_CATEGORIES.get(var, [])
        synonyms = VALUE_SYNONYM_MAP.get(var, {})
        data[var] = clean_categorical_column(data[var], canonical, synonyms, column_name=var)
    return data


def scale_continuous_column(series, center, scale):
    """
    Scale a continuous column using a FIXED, pre-specified anchor rather than a per-site
    fitted StandardScaler. A per-site fitted scaler would standardize each site's values
    relative to that site's own mean/SD, giving the two sites' fitted coefficients for
    this feature different meanings once averaged together during federated aggregation.
    A fixed shared anchor keeps the same real-world meaning at both sites, while still
    not leaking any statistic computed from the data itself.
    Args:
        series (pd.Series): Raw continuous column.
        center (float): Fixed value to subtract (e.g. DIAGNOSIS_YEAR_CENTER).
        scale (float): Fixed value to divide by (e.g. DIAGNOSIS_YEAR_SCALE).
    Returns:
        pd.Series: Scaled column.
    """
    return (series.astype(float) - center) / scale


def fit_impute_train(train_df, columns, n_iter=5, random_state=42):
    """
    Chained-equations imputation, fit on the training split ONLY, producing a single
    completed training dataset.

    On each of `n_iter` passes over the columns, every column with missing values is
    re-imputed from a model fit on the OTHER columns' current (possibly
    previously-imputed-in-an-earlier-pass) values, using only rows where that column is
    itself observed to fit the model. Categorical columns use multinomial logistic
    regression; continuous columns use linear regression. This produces one completed
    dataset rather than multiple pooled-per-Rubin's-rules imputations.

    Note: statsmodels' MICEData was evaluated for this and rejected -- its imputation
    step only supports models that produce a single scalar prediction per row (e.g. OLS,
    binary Logit), and raises on `MNLogit`, which every categorical predictor here with
    more than 2 categories requires (MNLogit's fitted parameters and predictions are
    2-D, one value per category, which breaks both its Gaussian perturbation step and
    its predictive-mean-matching step, each of which expects a 1-D array). scikit-learn's
    classifiers handle multi-class targets natively with no such restriction.
    Args:
        train_df (pd.DataFrame): Training split, with missing values as NaN.
        columns (list): Columns to impute.
        n_iter (int): Number of chained-equations passes over the columns.
        random_state (int): Random seed for the initial fill and any stochastic steps.
    Returns:
        pd.DataFrame: Training split with `columns` imputed.
    """
    working = train_df.copy()
    if not working[columns].isna().any().any():
        return working

    rng = np.random.RandomState(random_state)
    is_categorical, categories_map, missing_mask = {}, {}, {}
    for col in columns:
        missing_mask[col] = working[col].isna()
        is_categorical[col] = not pd.api.types.is_numeric_dtype(working[col])
        if is_categorical[col]:
            categories_map[col] = sorted(working[col].dropna().unique().tolist())

    # Initial fill: a random draw from the observed distribution for categorical columns,
    # the median for continuous -- a standard MICE starting point, refined by the
    # chained-equations passes below.
    for col in columns:
        n_missing = int(missing_mask[col].sum())
        if n_missing == 0:
            continue
        observed = working.loc[~missing_mask[col], col]
        if is_categorical[col]:
            fill_values = rng.choice(observed.values, size=n_missing)
        else:
            fill_values = np.full(n_missing, observed.astype(float).median())
        working.loc[missing_mask[col], col] = fill_values

    encoders = {col: {cat: i for i, cat in enumerate(cats)} for col, cats in categories_map.items()}

    def _encode_others(df, exclude_col):
        # Predictor encoding for the imputation regressions themselves (distinct from the
        # final fixed-schema one-hot encoding done later): categorical columns are one-hot
        # encoded here, not integer-coded, since integer-coding a nominal variable (e.g.
        # RaceEthnicity) would impose a fake ordinal relationship between categories.
        # Continuous columns are standardized so their scale doesn't dominate/slow the
        # solver relative to 0/1 dummy columns.
        encoded = pd.DataFrame(index=df.index)
        for col in columns:
            if col == exclude_col:
                continue
            if is_categorical[col]:
                dummies = pd.get_dummies(
                    pd.Categorical(df[col], categories=categories_map[col]), prefix=col
                ).astype(float)
                dummies.index = df.index
                encoded = encoded.join(dummies)
            else:
                values = df[col].astype(float)
                std = values.std()
                encoded[col] = (values - values.mean()) / std if std > 0 else 0.0
        return encoded

    for _ in range(n_iter):
        for col in columns:
            if not missing_mask[col].any():
                continue
            encoded_others = _encode_others(working, col)
            fit_rows = ~missing_mask[col]
            X_fit = encoded_others.loc[fit_rows].values
            X_pred = encoded_others.loc[missing_mask[col]].values
            y_fit = working.loc[fit_rows, col]

            if is_categorical[col]:
                y_fit_codes = y_fit.map(encoders[col])
                if y_fit_codes.nunique() < 2:
                    continue  # nothing to learn from a single observed class; keep initial fill
                model = LogisticRegression(max_iter=1000)
                model.fit(X_fit, y_fit_codes)
                pred_codes = model.predict(X_pred)
                inv_map = {i: cat for cat, i in encoders[col].items()}
                working.loc[missing_mask[col], col] = [inv_map[c] for c in pred_codes]
            else:
                model = LinearRegression()
                model.fit(X_fit, y_fit.astype(float))
                working.loc[missing_mask[col], col] = model.predict(X_pred)

    return working


def compute_test_fill_reference(imputed_train_df, columns):
    """
    Compute per-column reference fill values (mode for categorical, median for
    continuous) from the ALREADY-IMPUTED training split, for later use filling any
    residual missingness in the test split.

    The test split intentionally uses this simpler single-value fill rather than
    fitting its own chained-equations model: doing so keeps the test split's imputation
    informed exclusively by training-derived values, with no possibility of test-set
    information leaking into how any value (train or test) gets filled.
    """
    reference_values = {}
    for col in columns:
        if pd.api.types.is_numeric_dtype(imputed_train_df[col]):
            reference_values[col] = float(imputed_train_df[col].median())
        else:
            reference_values[col] = imputed_train_df[col].mode(dropna=True).iloc[0]
    return reference_values


def impute_test_with_reference(test_df, columns, reference_values):
    """
    Fill any missing values in the test split using ONLY training-derived reference
    values (see compute_test_fill_reference()) -- nothing is fit on the test split
    itself, which is the leakage this restructure exists to prevent.
    """
    working = test_df.copy()
    for col in columns:
        if working[col].isna().any():
            working[col] = working[col].fillna(reference_values[col])
    return working


def one_hot_encode_fixed_schema(data, categorical_vars):
    """
    One-hot encode each categorical column using the FIXED canonical category schema
    (constants.CANONICAL_CATEGORIES), rather than running pd.get_dummies() independently
    per split. This guarantees train and test always produce identical dummy columns
    regardless of which categories happen to appear in either split, replacing the
    previous assert-based fragility (`assert 'CCI_None' in all_cols`, etc.) with a
    structural guarantee.

    For any variable whose canonical category list is not yet populated in
    constants.CANONICAL_CATEGORIES (currently the BMC-local variables InsuranceType,
    Housing, LanguageGroup -- pending confirmation), this falls back to ordinary
    data-driven pd.get_dummies() for that column and emits a warning, rather than
    silently dropping the variable.
    Args:
        data (pd.DataFrame): Cleaned, imputed data (train or test split).
        categorical_vars (list): Categorical variable names to encode.
    Returns:
        pd.DataFrame: Data with categorical columns replaced by one-hot columns.
    """
    encoded = data.copy()
    for var in categorical_vars:
        if var not in encoded.columns:
            continue
        canonical = CANONICAL_CATEGORIES.get(var, [])
        if not canonical:
            warnings.warn(
                f"No canonical category list defined for '{var}' in "
                f"constants.CANONICAL_CATEGORIES; falling back to data-driven one-hot "
                f"encoding for this column, which does NOT guarantee identical columns "
                f"across splits/sites. Populate CANONICAL_CATEGORIES['{var}'] once its "
                f"category list is confirmed."
            )
            dummies = pd.get_dummies(encoded[var], prefix=var, drop_first=False)
            encoded = encoded.drop(columns=[var]).join(dummies)
            continue

        encoded[var] = pd.Categorical(encoded[var], categories=canonical)
        dummies = pd.get_dummies(encoded[var], prefix=var, drop_first=False)
        ordered_cols = [f"{var}_{cat}" for cat in canonical]
        for col_name in ordered_cols:
            if col_name not in dummies.columns:
                dummies[col_name] = False
        encoded = encoded.drop(columns=[var]).join(dummies[ordered_cols])
    return encoded


def stratify_by_diagnosis_era(data, diagnosis_year_col="DiagnosisYear", cutpoint=DIAGNOSIS_ERA_CUTPOINT):
    """
    Split a dataframe into pre-/post-cutpoint diagnosis-year strata, for a stratified
    sensitivity analysis by treatment era. NOTE: expects the RAW (unscaled) DiagnosisYear
    value -- call this before scale_continuous_column(), or pass the already-unscaled
    year explicitly.

    Not wired into the main preprocessing flow above: this is a standalone analysis
    helper to be run once DiagnosisYear is available in the raw data, comparing model
    performance separately pre- and post-cutpoint rather than altering the primary
    preprocessing pipeline.
    Args:
        data (pd.DataFrame): Data containing an unscaled `diagnosis_year_col`.
        diagnosis_year_col (str): Column name holding the raw diagnosis year.
        cutpoint (int): Year at/after which a row falls in the "later" stratum.
    Returns:
        tuple: (pre_cutpoint_df, post_cutpoint_df)
    """
    pre = data[data[diagnosis_year_col] < cutpoint].reset_index(drop=True)
    post = data[data[diagnosis_year_col] >= cutpoint].reset_index(drop=True)
    return pre, post


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preprocessing Script")

    parser.add_argument("--env_path", type=str, default=".env",
                        help="Path to the environment file containing data paths.")
    parser.add_argument("--data_source", type=str, default="va",
                        help="Data source identifier (e.g., 'va' or 'bmc').")
    parser.add_argument("--test_size", type=float, default=0.4,
                        help="Test-set fraction of the 2-way train/test split. Default 0.4 (60/40 train/test).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for the train/test split (and everything else, via set_seed).")

    args = parser.parse_args()
    set_seed(args.seed)
    load_dotenv(args.env_path)
    file_path = os.getenv("RAW_DATA_PATH")
    save_dir = os.getenv("DATA_DIR")
    if not file_path or not save_dir:
        missing = [name for name, val in [("RAW_DATA_PATH", file_path), ("DATA_DIR", save_dir)] if not val]
        raise ValueError(
            f"Missing required environment variable(s) {missing} after loading '{args.env_path}'. "
            f"Check that this file exists (relative to your current working directory -- run this "
            f"script from inside FedMann/) and defines both RAW_DATA_PATH and DATA_DIR, one per "
            f"line, with no surrounding quotes."
        )
    data_source = args.data_source.lower()
    if data_source == "va":
        LOCAL_VARS = LOCAL_VARS_VA
    elif data_source == "bmc":
        LOCAL_VARS = LOCAL_VARS_BMC
    else:
        raise ValueError("--data_source must be 'va' or 'bmc'.")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    categorical_vars = SHARED_VARS + LOCAL_VARS
    target_var = TARGET_COL

    # Load raw data. Column names are assumed already harmonized across sites (VA and BMC
    # use identical predictor names) -- this is a manual, human-reviewed step done before
    # running this script, not auto-detected here.
    raw_data = pd.read_csv(file_path, dtype={"CCI": str})

    # DiagnosisYear is optional/guarded until the VA/BMC re-pull that includes it lands --
    # everything else in SHARED_VARS/LOCAL_VARS is required.
    continuous_vars = [v for v in CONTINUOUS_VARS if v in raw_data.columns]
    missing_continuous = [v for v in CONTINUOUS_VARS if v not in continuous_vars]
    if missing_continuous:
        print(f"Note: continuous predictor(s) {missing_continuous} not found in raw data yet "
              f"-- proceeding without them.")

    feature_vars = categorical_vars + continuous_vars
    required_cols = feature_vars + [target_var]
    missing_cols = [c for c in required_cols if c not in raw_data.columns]
    if missing_cols:
        raise KeyError(
            f"Expected column(s) {missing_cols} not found in {file_path}. Raw column names "
            f"must be pre-harmonized to the canonical names in constants.py before running "
            f"this script."
        )

    data = raw_data[feature_vars + [target_var]].copy()

    # The outcome must be fully observed to train/evaluate on a row; this drops rows with
    # no recorded outcome only -- it does NOT drop rows for missing predictor values
    # (those are imputed below instead of discarded).
    n_before = len(data)
    data = data.dropna(subset=[target_var]).reset_index(drop=True)
    if len(data) < n_before:
        print(f"Dropped {n_before - len(data)} row(s) with a missing outcome value.")
    data[target_var] = data[target_var].astype(int)

    # Clean/validate categorical values (whitespace, casing, known synonyms, fail-fast on
    # anything unrecognized) BEFORE anything else touches them.
    data = clean_dataframe(data, categorical_vars)

    # Split into train/test FIRST, BEFORE imputation or encoding are fit, so that neither
    # is informed by any information from the held-out test rows.
    train_data, test_data = train_test_split(data, test_size=args.test_size, random_state=args.seed)

    # Fit imputation on the training split only; the test split is filled using
    # training-derived reference values only (see compute_test_fill_reference() doc).
    impute_cols = categorical_vars + continuous_vars
    train_data = fit_impute_train(train_data, impute_cols)
    reference_values = compute_test_fill_reference(train_data, impute_cols)
    test_data = impute_test_with_reference(test_data, impute_cols, reference_values)

    # Scale the continuous DiagnosisYear predictor (if present) using a fixed, shared
    # anchor -- identical at both sites -- not a per-site fitted scaler.
    for col in continuous_vars:
        train_data[col] = scale_continuous_column(train_data[col], DIAGNOSIS_YEAR_CENTER, DIAGNOSIS_YEAR_SCALE)
        test_data[col] = scale_continuous_column(test_data[col], DIAGNOSIS_YEAR_CENTER, DIAGNOSIS_YEAR_SCALE)

    # One-hot encode categorical variables using the fixed canonical schema, so train and
    # test always share identical columns regardless of which categories appear in either.
    train_data = one_hot_encode_fixed_schema(train_data, categorical_vars)
    test_data = one_hot_encode_fixed_schema(test_data, categorical_vars)

    feature_cols = sorted([c for c in train_data.columns if c != target_var])  # sort is important for downstream tracking
    local_cols = get_local_cols(feature_cols, LOCAL_VARS)
    meta_data = {
        "feature_cols": feature_cols,
        "local_cols": local_cols,
        "target_col": target_var,
        "continuous_cols": continuous_vars,
        "split": {"design": "train_test_2way", "test_size": args.test_size, "seed": args.seed},
    }
    with open(f"{save_dir}/meta.json", "w") as f:
        json.dump(meta_data, f)

    train_data.to_csv(f"{save_dir}/train.csv", index=False)
    test_data.to_csv(f"{save_dir}/test.csv", index=False)

    # Save the fitted imputation reference values and the canonical schema alongside
    # meta.json so the exact preprocessing can be reapplied to new data.
    with open(f"{save_dir}/preprocessing_artifacts.pkl", "wb") as f:
        pickle.dump({
            "imputation_reference_values": reference_values,
            "canonical_categories": CANONICAL_CATEGORIES,
            "value_synonym_map": VALUE_SYNONYM_MAP,
            "seed": args.seed,
        }, f)

    print(f"Saved {len(train_data)} training and {len(test_data)} test rows to {save_dir} "
          f"({len(feature_cols)} feature columns).")
