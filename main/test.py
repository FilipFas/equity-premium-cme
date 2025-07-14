"""
main.test.py

This script evaluates the out-of-sample performance of the 
Conditional Mean Embedding (CME) model using the best 
hyperparameters identified during tuning. It performs:

    1) Data ingestion and lag creation
    2) Rolling-window out-of-sample testing
    3) Statistical comparison to benchmark forecasts
    4) Aggregation and persistence of test performance metrics

Author: Filippo Fasoli
"""

import pandas as pd
import numpy as np
import os, ast
from scipy.stats import f, norm
import statsmodels.api as sm

# Conditional Mean Embedding implementation for nonparametric forecasting
from cme import ConditionalMeanEmbedding
# Utilities for data preprocessing and rolling-window generation
from cleaning import RollingWindowSplitter, DataCleaner
# Configuration constants: feature sets, kernel choices, split parameters, and file paths
from config import FEATURE_SELECTION, FIXED_KERNEL, ROLLING_SPLIT_PARAMS, DATA_PATHS

# -----------------------------------------------------------------------------
# Step 1: Load validation results
# -----------------------------------------------------------------------------
# These results (produced by main_train.py) contain the optimal hyperparameters
# selected for each frequency based on validation R² performance.
VAL_CSV = f"{FIXED_KERNEL}_multivar_train_results.csv"

# Ensure the file exists; raise an error otherwise
if not os.path.exists(VAL_CSV):
    raise FileNotFoundError(f"{VAL_CSV} not found. Run main_train.py first.")
val_df = pd.read_csv(VAL_CSV)

# -----------------------------------------------------------------------------
# Step 2: Load cleaned datasets and apply feature selection
# -----------------------------------------------------------------------------
# Ingest preprocessed CSVs, rename target to 'y', and subset features
df_ann = pd.read_csv(DATA_PATHS["annual"]).rename(columns={'equity_premium_annual': 'y'})
df_ann = df_ann[FEATURE_SELECTION["annual"]["keep"]]

df_qtr = pd.read_csv(DATA_PATHS["quarterly"]).rename(columns={'equity_premium_quarterly': 'y'})
df_qtr = df_qtr[FEATURE_SELECTION["quarterly"]["keep"]]

df_mon = pd.read_csv(DATA_PATHS["monthly"]).rename(columns={'equity_premium_monthly': 'y'})
df_mon = df_mon[FEATURE_SELECTION["monthly"]["keep"]]

# -----------------------------------------------------------------------------
# Step 3: Generate lagged features for dynamic forecasting
# -----------------------------------------------------------------------------
# Create autoregressive structures using p lags of predictors and q lags of the target
df_ann_dyn = DataCleaner.make_lags(df_ann, p=4, q=1)
df_qtr_dyn = DataCleaner.make_lags(df_qtr, p=4, q=1)
df_mon_dyn = DataCleaner.make_lags(df_mon, p=4, q=1)

# -----------------------------------------------------------------------------
# Step 4: Construct rolling-window splits for each frequency
# -----------------------------------------------------------------------------
# These splits simulate an expanding out-of-sample test set
splits_map = {
    'annual': list(RollingWindowSplitter(df_ann_dyn, **ROLLING_SPLIT_PARAMS["annual"]).generate_splits()),
    'quarterly': list(RollingWindowSplitter(df_qtr_dyn, **ROLLING_SPLIT_PARAMS["quarterly"]).generate_splits()),
    'monthly': list(RollingWindowSplitter(df_mon_dyn, **ROLLING_SPLIT_PARAMS["monthly"]).generate_splits()),
}

# -----------------------------------------------------------------------------
# Step 5: Loop through frequencies and evaluate test performance
# -----------------------------------------------------------------------------
results = []

# Iterate through each row of validation results (one per frequency)
for _, row in val_df.iterrows():
    freq = row['frequency']
    best_p = ast.literal_eval(row['best_params'])  # Parse stringified dict
    splits = splits_map[freq]

    # Extract model hyperparameters
    lam = best_p['lambda_reg']
    kt  = row.get("kernel", FIXED_KERNEL)
    poly_degree = best_p.get('poly_degree')
    poly_gamma  = best_p.get('poly_gamma')
    poly_coef0  = best_p.get('poly_coef0')

    # Handle separate treatment of ARD-RBF kernel length-scales
    if kt == "ard-rbf":
        keys = sorted([k for k in best_p if k.startswith("ard_ls_")],
                      key=lambda s: int(s.split('_')[-1]))
        ls = [best_p[k] for k in keys]
    else:
        ls = best_p['length_scale']

    # Containers for stacking results across rolling splits
    all_y, all_yhat, all_bench = [], [], []

    # -----------------------------------------------------------------------------
    # Step 5a: Rolling-window out-of-sample forecasting
    # -----------------------------------------------------------------------------
    for train, val, test in splits:
        # Merge train and validation for final model training
        X_train = pd.concat([train.drop(columns='y'), val.drop(columns='y')]).values
        y_train = np.concatenate([train['y'].values, val['y'].values])
        X_test = test.drop(columns='y').values
        y_test = test['y'].values

        # Fit CME model on combined training data
        model = ConditionalMeanEmbedding(
            kernel_type=kt,
            length_scale=ls,
            lam=lam,
            poly_degree=poly_degree,
            poly_gamma=poly_gamma,
            poly_coef0=poly_coef0
        )
        model.fit(X_train, y_train)
        yhat = model.predict(X_test)

        # Compute benchmark forecast using historical mean
        history = list(y_train)
        bench = []
        for i in range(len(y_test)):
            bench.append(np.mean(history + y_test[:i].tolist()))

        # Store true values, predictions, and benchmarks
        all_y.append(y_test)
        all_yhat.append(yhat)
        all_bench.append(np.array(bench))

    # -----------------------------------------------------------------------------
    # Step 6: Aggregate and evaluate performance metrics
    # -----------------------------------------------------------------------------
    y = np.concatenate(all_y)
    yhat = np.concatenate(all_yhat)
    bench = np.concatenate(all_bench)

    # === OOS R² ===
    e_c = y - yhat
    e_b = y - bench
    r2_oos = 1 - np.sum(e_c**2) / np.sum(e_b**2)

    # === MSE-F Statistic (Clark-West) ===
    q = 1
    P = len(y)
    MSE_b = np.mean(e_b**2)
    MSE_c = np.mean(e_c**2)
    F_stat = ((MSE_c - MSE_b) / MSE_b) * ((P - q) / q)
    p_F = 1 - f.cdf(F_stat, dfn=q, dfd=P - q)

    # === OOSCT t-statistic (West-McCracken) ===
    d = e_b**2 - e_c**2
    H = 4  # Newey-West HAC lags
    mdl = sm.OLS(d, np.ones_like(d))
    res = mdl.fit(cov_type='HAC', cov_kwds={'maxlags': H})
    t_oos = res.params[0] / res.bse[0]
    p_t = 1 - norm.cdf(t_oos)

    # Helper to format R² with significance stars
    def star(r2, p):
        if p < 0.01:
            sig = '***'
        elif p < 0.05:
            sig = '**'
        elif p < 0.10:
            sig = '*'
        else:
            sig = ''
        return f"{r2:.3f}{sig}"

    # Store results for current frequency
    results.append({
        "frequency": freq,
        "test_R2": star(r2_oos, p_t),
        "MSE_F": F_stat,
        "p_MSE_F": p_F,
        "OOSCT_t": t_oos,
        "p_OOSCT_t": p_t
    })

# -----------------------------------------------------------------------------
# Step 7: Save results to disk
# -----------------------------------------------------------------------------
# Save the test results in a structured CSV file for future inspection or reporting
out_csv = f"TEST_{FIXED_KERNEL}_multivar_results.csv"
pd.DataFrame(results).to_csv(out_csv, index=False)
print(f"\n✅ Test results saved to {out_csv}")
