"""
main.train.py

This script orchestrates the end-to-end workflow for:
    1) Cleaning and preprocessing time‐series data at multiple frequencies
    2) Tuning the Covariate‐Adjusted Mean Estimator (CME) model via Optuna
    3) Aggregating and persisting performance metrics for subsequent analysis

Author: Filippo Fasoli
"""

import pandas as pd
import optuna

# Custom modules for data cleaning and model evaluation
from cleaning import MultiFrequencyCleaner
from cme import optuna_objective
from config import CLEANING_CONFIG, OPTUNA_TRIALS, FIXED_KERNEL

# -----------------------------------------------------------------------------
# Step 1: Execute the full data cleaning pipeline
# -----------------------------------------------------------------------------
# Instantiate the cleaning class with user-defined parameters.
# This will handle missing values, feature engineering, and
# alignment of multiple time‐series frequencies.
cleaner = MultiFrequencyCleaner(CLEANING_CONFIG)
cleaner.run_all()

# After cleaning is complete, retrieve the train/validation/test splits
# organized by frequency: annual, quarterly, and monthly.
splits = cleaner.get_splits()

# -----------------------------------------------------------------------------
# Step 2: Hyperparameter tuning of the CME model for each frequency
# -----------------------------------------------------------------------------
results = []

for freq in ['annual', 'quarterly', 'monthly']:
        print(f"\n🔧 Tuning CME for {freq.upper()} frequency...")
        
        # Select the appropriate data split for the current frequency
        split_list = splits[freq]
        
        # Create an Optuna study to maximize the R² metric
        # The `optuna_objective` encapsulates:
        #   - fitting the model on the training fold
        #   - evaluating on the validation fold
        #   - returning the R² score as the optimization target
        study = optuna.create_study(direction="maximize")
        study.optimize(
                lambda trial: optuna_objective(trial, split_list, FIXED_KERNEL),
                n_trials=OPTUNA_TRIALS
        )

        # Extract the best performing hyperparameters and associated R²
        best_params = study.best_params
        best_r2     = study.best_value

        # Record the results for later aggregation
        results.append({
                "frequency":   freq,
                "kernel":      FIXED_KERNEL,
                "pooled_R2":   best_r2,
                "best_params": best_params
        })

        print(f"✅ Done tuning {freq}. Best R²: {best_r2:.3f}")

# -----------------------------------------------------------------------------
# Step 3: Persist the tuning results for downstream analysis
# -----------------------------------------------------------------------------
# Convert the list of dicts into a pandas DataFrame, then save as CSV.
# This file can be used for reporting, visualization, or further validation.
val_df = pd.DataFrame(results)
output_filename = f"{FIXED_KERNEL}_multivar_train_results.csv"
val_df.to_csv(output_filename, index=False)

print(f"\n✅ All frequencies tuned. Validation results saved to {output_filename}.")
