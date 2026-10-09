
# Dataset constants
TARGET_COL = 'NoTripletOrTransplant'
SHARED_VARS = ["KmFromHospital", "SmokingStatus", "RaceEthnicity", "AgeGroup", "Gender",
                    "CCI", "StageBest"]
LOCAL_VARS_VA = ['ADINationalRank', 'Income']
LOCAL_VARS_BMC = ['InsuranceType', 'Housing', 'LanguageGroup']

# Continuous predictors are handled separately from the categorical SHARED_VARS/LOCAL_VARS
# above: they are not one-hot encoded, and are scaled using a fixed, shared anchor (not a
# per-site fitted scaler) so that the same raw value has the same meaning at every site,
# which matters for federated parameter averaging (see DIAGNOSIS_YEAR_CENTER/SCALE below).
CONTINUOUS_VARS = ["DiagnosisYear"]

# Fixed (not data-derived) scaling constants for DiagnosisYear: (value - CENTER) / SCALE.
# 2017 is the study-period midpoint (2012-2022); 5 is a rough half-decade scale. Using a
# fixed, pre-specified anchor rather than a per-site StandardScaler keeps this feature's
# meaning identical across sites, so that averaging its fitted coefficient during federated
# aggregation remains coherent.
DIAGNOSIS_YEAR_CENTER = 2017
DIAGNOSIS_YEAR_SCALE = 5

# Cutpoint for the diagnosis-year stratified sensitivity analysis: 2019 is when
# daratumumab-based quadruplet regimens entered front-line use (GRIFFIN and related trials),
# which is directly relevant to the "guideline-concordant primary therapy" outcome definition,
# since quadruplet is one of its qualifying regimens and wasn't a realistic option before then.
DIAGNOSIS_ERA_CUTPOINT = 2019

# Full, pre-specified set of valid canonical categories for each categorical variable. This is
# the single source of truth for BOTH (a) validating/cleaning raw category values before
# imputation, and (b) the fixed one-hot encoding schema applied identically to every split, so
# the two can never drift out of sync. Extend this list if a new legitimate category is added
# to the raw data (e.g., a new CCI band); do not add a category here just to silence a cleaning
# error without confirming the underlying raw value is legitimate.
CANONICAL_CATEGORIES = {
    "AgeGroup": ["<55", "55-65", "65-75", "75-85", "85+"],
    "Gender": ["F", "M"],
    "RaceEthnicity": ["Hispanic", "Non-Hispanic Black", "Non-Hispanic White", "Other"],
    "SmokingStatus": ["Current", "Former", "Never"],
    "CCI": ["None", "Mild", "Moderate", "Severe"],
    "StageBest": ["1", "2", "3"],
    "KmFromHospital": ["<25", "25-50", "50-75", "75-100", "100-125", "125+"],
    # BMC-local; populated from the observed values in the extended BMC export
    # (bmc_extended_with_dxyear.csv). "Unknown" is deliberately excluded here -- see
    # VALUE_SYNONYM_MAP below, it is treated as missing (to be imputed), not as its own
    # category, consistent with RaceEthnicity/CCI/StageBest.
    "InsuranceType": ["Private", "Public", "Self Pay"],
    "Housing": ["Domiciled", "Undomiciled"],
    "LanguageGroup": ["English", "Spanish", "Haitian Creole", "Other"],
    # VA-local; national ADI rank quintiles (First=least disadvantaged, 1-20, ... Fifth=most
    # disadvantaged, 81-100) and income brackets, as given.
    "ADINationalRank": ["First (1-20)", "Second (21-40)", "Third (41-60)", "Fourth (61-80)", "Fifth (81-100)"],
    "Income": ["<25K", "25K-50K", "50K-75K", "75K-100K", "100K+"],
}

# Known messy-value synonyms per variable, mapped to their canonical form. This starts small
# and should grow as real messy values are found in the raw data (see
# clean_categorical_column() in data_prep.py) -- entries here are applied AFTER whitespace
# stripping and casing normalization, so keys should already be lowercase/stripped. A mapped
# value of None means the raw value is genuinely missing (to be imputed), not a variant
# spelling of a real category -- e.g. CCI's "unknown"/"unk"/"n/a" are distinct from the
# "None" comorbidity band (CCI=0), which Table 1 reports as its own separate category, so
# they must not be folded into "None" as if they were a real score.
VALUE_SYNONYM_MAP = {
    "CCI": {
        "unk": None,
        "n/a": None,
        "unknown": None,
    },
    "RaceEthnicity": {
        "hispanic or latino": "Hispanic",
        "unknown": None,
    },
    # BMC's extended export labels age bins as non-overlapping decades (e.g. "55-64")
    # rather than the "55-65" convention already used elsewhere; same bins, different
    # boundary labeling.
    "AgeGroup": {
        "55-64": "55-65",
        "65-74": "65-75",
        "75-84": "75-85",
        "unknown": None,
    },
    "Gender": {
        "male": "M",
        "female": "F",
        "unknown": None,
    },
    "SmokingStatus": {
        "unknown": None,
    },
    "StageBest": {
        "unknown": None,
    },
    "InsuranceType": {
        "unknown": None,
    },
    "Housing": {
        "unknown": None,
    },
    "LanguageGroup": {
        "unknown": None,
    },
    "KmFromHospital": {
        "unknown": None,
    },
    # "First(1-20)" has no space before the parenthesis, unlike the other four quintile
    # labels ("Second (21-40)", etc.) -- mapped explicitly since the cleaning step only
    # auto-normalizes case/whitespace-trimming, not internal spacing differences.
    "ADINationalRank": {
        "first(1-20)": "First (1-20)",
        "unknown": None,
    },
    "Income": {
        "unknown": None,
    },
}

# Model location constants
CENTRAL_MODEL_BASE = 'model/central_model'
CENTRAL_MODEL_BAK = 'model/central_model_bak'
NODE_MODEL_BASE = 'model/node_model'
NODE_MODEL_LOCAL_BASE = 'model/node_model_local'
NODE_MODEL_LOCAL_SHARED_OTHER_BAK = 'model/node_model_local_shared_other_bak'
NODE_MODEL_LOCAL_SHARED_BASE = 'model/node_model_local_shared'
NODE_MODEL_LOCAL_FULL_BASE = 'model/node_model_local_full'
