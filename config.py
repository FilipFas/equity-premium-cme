"""
config.py

This module provides centralized configuration for the thesis workflow, including:
    1) FIXED_KERNEL: choice of kernel function for the CME model
    2) FEATURE_SELECTION: variables to keep or drop for annual, quarterly, and monthly datasets
    3) ROLLING_SPLIT_PARAMS: rolling window cross-validation parameters (train, val, test sizes and step)
    4) DATA_PATHS: filenames for cleaned time-series data at different frequencies
    5) CLEANING_CONFIG: raw data filepaths, output directories, transformation (log/std) settings, and date filters
    6) OPTUNA_TRIALS: number of trials for hyperparameter tuning with Optuna

Author: Filippo Fasoli
"""

# -----------------------------------------------------------------------------
# Kernel configuration
# -----------------------------------------------------------------------------
# This string defines the fixed kernel to use throughout the workflow.
# Supported values: "rbf", "laplacian", "ard-rbf", "matern", "polynomial"
FIXED_KERNEL = "polynomial"

# -----------------------------------------------------------------------------
# Feature selection: variables to retain or discard per frequency
# -----------------------------------------------------------------------------
# These are based on domain knowledge and empirical performance.
# "keep" includes predictors and the target variable ('y'), 
# while "drop" lists columns to exclude before lag creation.
FEATURE_SELECTION = {
    "annual": {
        "keep": ["y", "eqis", "ltr", "corpr", "infl", "cay", "fbm", "dtoy", "rdsp", "gip", "tchi"],
        "drop": ["yyyy", "Rfree", "ret"]
    },
    "quarterly": {
        "keep": ["y", "pce", "crdstd", "i/k", "shtint", "b/m", "vp", "dtoy"],
        "drop": ["yyyyq", "Rfree", "ret"]
    },
    "monthly": {
        "keep": ["y", "ygap", "tbl", "skvw", "tail", "dtoy", "dtoat", "lty"],
        "drop": ["yyyymm", "Rfree", "ret"]
    }
}

# -----------------------------------------------------------------------------
# Rolling window cross-validation parameters
# -----------------------------------------------------------------------------
# Defines the size of training, validation, and test sets, 
# along with the step size for the rolling split.
ROLLING_SPLIT_PARAMS = {
    "annual":    dict(train_size=30,  val_size=10,  test_size=10,  step_size=1),
    "quarterly": dict(train_size=80,  val_size=30,  test_size=10,  step_size=5),
    "monthly":   dict(train_size=800, val_size=200, test_size=100, step_size=25)
}

# -----------------------------------------------------------------------------
# Filepaths to cleaned datasets
# -----------------------------------------------------------------------------
# These CSV files are generated after the preprocessing stage.
# Used as input in the training and testing pipelines.
DATA_PATHS = {
    "annual":   "data_annual_cleaned.csv",
    "quarterly":"data_quarterly_cleaned.csv",
    "monthly":  "data_monthly_cleaned.csv"
}

# -----------------------------------------------------------------------------
# Cleaning configuration for each frequency
# -----------------------------------------------------------------------------
# Specifies raw data locations, transformation settings, and date filtering rules.
# Used by the MultiFrequencyCleaner to instantiate the DataCleaner objects.
CLEANING_CONFIG = {
    'annual': {
        'filepath': '/Users/FilippoFasoli_1/Desktop/THESIS/MAIN/Data2023_annual.csv',
        'output_dir': '/Users/FilippoFasoli_1/Desktop/THESIS/MAIN/',
        'log_vars': ['cfacc', 'eqis'],  # Apply log transform
        'std_vars': ['accrul', 'gpce', 'cfacc_log', 'eqis_log'],  # Standardize
        'date_col': 'yyyy',  # Time column used for filtering
        'start_date': 1965   # First year to include
    },
    'quarterly': {
        'filepath': '/Users/FilippoFasoli_1/Desktop/THESIS/MAIN/Data2023_quarterly.csv',
        'output_dir': '/Users/FilippoFasoli_1/Desktop/THESIS/MAIN/',
        'std_vars': ['pce', 'crdstd', 'i/k'],  # Variables to standardize
        'date_col': 'yyyyq',
        'start_date': 19902  # First quarter (YYYYQ format)
    },
    'monthly': {
        'filepath': '/Users/FilippoFasoli_1/Desktop/THESIS/MAIN/Data2023_monthly.csv',
        'output_dir': '/Users/FilippoFasoli_1/Desktop/THESIS/MAIN/',
        'log_vars': ['tbl', 'lty'],
        'std_vars': ['tbl_log', 'lty_log'],
        'date_col': 'yyyymm',
        'start_date': 192601  # First month (YYYYMM format)
    }
}

# -----------------------------------------------------------------------------
# Optuna hyperparameter tuning settings
# -----------------------------------------------------------------------------
# Number of optimization trials to perform for each frequency
OPTUNA_TRIALS = 150
