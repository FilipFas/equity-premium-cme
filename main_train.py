import pandas as pd
import optuna
from cleaning import MultiFrequencyCleaner
from cme import ConditionalMeanEmbedding, optuna_objective

# 1) Define kernel to use for all frequencies (can be passed via argparse or config)
FIXED_KERNEL = "polynomial"

# 2) Define cleaning + lag + rolling config per frequency
config = {
    'annual': {
        'filepath': '/Your_Path/Data2023_annual.csv',
        'output_dir': '/Your_Path',
        'log_vars': ['cfacc', 'eqis'],
        'std_vars': ['accrul', 'gpce', 'cfacc_log', 'eqis_log'],
        'date_col': 'yyyy',
        'start_date': 1965
    },
    'quarterly': {
        'filepath': '/Your_Path/Data2023_quarterly.csv',
        'output_dir': '/Your_Path',
        'std_vars': ['pce', 'crdstd', 'i/k'],
        'date_col': 'yyyyq',
        'start_date': 19902
    },
    'monthly': {
        'filepath': '/Your_Path/Data2023_monthly.csv',
        'output_dir': '/Your_Path',
        'log_vars': ['tbl', 'lty'],
        'std_vars': ['tbl_log', 'lty_log'],
        'date_col': 'yyyymm',
        'start_date': 192601
    }
}

# 3) Run full data cleaning pipeline
cleaner = MultiFrequencyCleaner(config)
cleaner.run_all()
splits = cleaner.get_splits()

# 4) Tune CME model for each frequency
results = []
for freq in ['annual', 'quarterly', 'monthly']:
    print(f"\n🔧 Tuning CME for {freq.upper()} frequency...")
    split_list = splits[freq]

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: optuna_objective(t, split_list, FIXED_KERNEL), n_trials=100)

    best_params = study.best_params
    best_r2 = study.best_value

    # Save results
    results.append({
        "frequency": freq,
        "kernel": FIXED_KERNEL,
        "pooled_R2": best_r2,
        "best_params": best_params
    })

    print(f"✅ Done tuning {freq}. Best R²: {best_r2:.3f}")

# 5) Save validation results to CSV

val_df = pd.DataFrame(results)
val_df.to_csv(f"{FIXED_KERNEL}_multivar_train_results.csv", index=False)
print("\n✅ All frequencies tuned. Validation results saved.")
