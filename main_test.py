
import pandas as pd
import numpy as np
import os, ast
from sklearn.metrics import mean_squared_error
from scipy.stats import f, norm
import statsmodels.api as sm
from cme import ConditionalMeanEmbedding
from cleaning import RollingWindowSplitter, DataCleaner

# === CONFIG ===
FIXED_KERNEL = "polynomial"
VAL_CSV = f"{FIXED_KERNEL}_multivar_train_results.csv"

# === Load validation results ===
if not os.path.exists(VAL_CSV):
    raise FileNotFoundError(f"{VAL_CSV} not found. Run main_train.py first.")
val_df = pd.read_csv(VAL_CSV)

# === Load cleaned datasets ===
df_ann = pd.read_csv("data_annual_cleaned.csv").rename(columns={'equity_premium_annual':'y'})[
    ["y","eqis","ltr","corpr","infl","cay","fbm","dtoy","rdsp","gip","tchi"]
]
df_qtr = pd.read_csv("data_quarterly_cleaned.csv").rename(columns={'equity_premium_quarterly':'y'})[
    ["y","pce","crdstd","i/k","shtint","b/m","vp","dtoy"]
]
df_mon = pd.read_csv("data_monthly_cleaned.csv").rename(columns={'equity_premium_monthly':'y'})[
    ["y","ygap","tbl","skvw","tail","dtoy","dtoat","lty"]
]

# === Create lags ===
df_ann_dyn = DataCleaner.make_lags(df_ann, p=4, q=1)
df_qtr_dyn = DataCleaner.make_lags(df_qtr, p=4, q=1)
df_mon_dyn = DataCleaner.make_lags(df_mon, p=4, q=1)

# === Rolling splits ===
splits_map = {
    'annual': list(RollingWindowSplitter(df_ann_dyn, 30, 10, 10, 1).generate_splits()),
    'quarterly': list(RollingWindowSplitter(df_qtr_dyn, 80, 30, 10, 5).generate_splits()),
    'monthly': list(RollingWindowSplitter(df_mon_dyn, 800, 200, 100, 25).generate_splits()),
}

# === Testing Loop ===
results = []
for _, row in val_df.iterrows():
    freq = row['frequency']
    best_p = ast.literal_eval(row['best_params'])
    splits = splits_map[freq]

    # Extract CME hyperparameters
    lam = best_p['lambda_reg']
    kt  = row.get("kernel", FIXED_KERNEL)
    poly_degree = best_p.get('poly_degree')
    poly_gamma  = best_p.get('poly_gamma')
    poly_coef0  = best_p.get('poly_coef0')
    if kt == "ard-rbf":
        keys = sorted([k for k in best_p if k.startswith("ard_ls_")],
                      key=lambda s: int(s.split('_')[-1]))
        ls = [best_p[k] for k in keys]
    else:
        ls = best_p['length_scale']

    final_r2 = []
    all_y, all_yhat, all_bench = [], [], []

    for train, val, test in splits:
        X_train = pd.concat([train.drop(columns='y'), val.drop(columns='y')]).values
        y_train = np.concatenate([train['y'].values, val['y'].values])
        X_test = test.drop(columns='y').values
        y_test = test['y'].values

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

        # Prevailing-mean benchmark
        history = list(y_train)
        bench = []
        for i in range(len(y_test)):
            bench.append(np.mean(history + y_test[:i].tolist()))

        all_y.append(y_test)
        all_yhat.append(yhat)
        all_bench.append(np.array(bench))

    y = np.concatenate(all_y)
    yhat = np.concatenate(all_yhat)
    bench = np.concatenate(all_bench)

    # === Evaluation ===
    e_c = y - yhat
    e_b = y - bench

    # R2
    r2_oos = 1 - np.sum(e_c**2) / np.sum(e_b**2)

    # MSE-F (Clark-West)
    q = 1
    P = len(y)
    MSE_b = np.mean(e_b**2)
    MSE_c = np.mean(e_c**2)
    F_stat = ((MSE_c - MSE_b)/MSE_b)*((P - q)/q)
    p_F = 1 - f.cdf(F_stat, dfn=q, dfd=P-q)

    # OOSCT t-stat
    d = e_b**2 - e_c**2
    H = 4
    mdl = sm.OLS(d, np.ones_like(d))
    res = mdl.fit(cov_type='HAC', cov_kwds={'maxlags': H})
    t_oos = res.params[0] / res.bse[0]
    p_t = 1 - norm.cdf(t_oos)

    def star(r2, p):
        return f"{r2:.3f}" + ("***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "")

    results.append({
        "frequency": freq,
        "test_R2": star(r2_oos, p_t),
        "MSE_F": F_stat,
        "p_MSE_F": p_F,
        "OOSCT_t": t_oos,
        "p_OOSCT_t": p_t
    })

# === Save Results ===
out_csv = f"TEST_{FIXED_KERNEL}_multivar_results.csv"
pd.DataFrame(results).to_csv(out_csv, index=False)
print(f"\n✅ Test results saved to {out_csv}")
