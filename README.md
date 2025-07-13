# Enhancing Equity Premium Predictability with RKHS Conditional Mean Embeddings

(THIS IS STILL AN ONGOING PROJECT)

This repository contains the implementation of my Master Thesis project:

**Enhancing Equity Premium Predictability with Reproducing Kernel Hilbert Space Conditional Mean Embeddings**  
Supervised by **Prof. Paul Schneider**  
USI – Università della Svizzera italiana

---

## 📘 Overview

This project explores the use of **Conditional Mean Embeddings (CME)** in **Reproducing Kernel Hilbert Spaces (RKHS)** for forecasting the **equity premium** using macro-financial predictors. The method offers a flexible, theoretically grounded alternative to linear and machine learning models by nonparametrically estimating conditional expectations.

---

## 🧠 Key Concepts

- **Reproducing Kernel Hilbert Space (RKHS)**  
  A Hilbert space where evaluation at a point is a continuous linear functional, enabling non-linear modeling via kernel functions.

- **Conditional Mean Embedding (CME)**  
  A nonparametric representation of the conditional expectation \( \mathbb{E}[Y \mid X = x] \) in an RKHS.

- **Rolling Window Evaluation**  
  Out-of-sample testing using realistic rolling training/validation/test splits.

- **Hyperparameter Optimization with Optuna**  
  Regularization and kernel parameters are optimized using a pooled out-of-sample \( R^2 \) objective.

---
## 📁 Project Structure
```bash
├── cme.py        # Core CME logic and kernel support
├── cleaning.py         # Data preprocessing, lags, rolling windows
├── main_train.py       # Hyperparameter tuning via Optuna
├── main_test.py        # Final testing using best config
├── results/            # Output CSVs for validation and test
├── data/               # Input macro-financial data (external)
└── README.md           # Project documentation
```


---

## ⚙️ Requirements

- Python ≥ 3.9
- `numpy`
- `pandas`
- `scikit-learn`
- `optuna`
- `statsmodels`
- `scipy`

Install dependencies with:

```bash
pip install -r requirements.txt
```
## 📊 Evaluation Metrics

The model is evaluated using:

- **Out-of-Sample \( R^2 \)** (Campbell–Thompson)
- **MSE-F Statistic** (Clark–McCracken)
- **OOS t-statistic** with Newey–West HAC correction

---

## 📎 Notes

- The pipeline supports `monthly`, `quarterly`, and `annual` data frequencies.
- Feature engineering includes autoregressive lags and a rolling benchmark forecast.
- CME is tested across multiple kernel types and configurations.

---

## 📄 License

This repository is for academic and research use only.  
Contact the author for reuse in other contexts.

---

## 👤 Author

**Filippo Fasoli**  
MSc in Finance – USI Università della Svizzera italiana  
[LinkedIn](https://www.linkedin.com/in/filippo-fasoli/)  
📧 filippo.fasoli@usi.ch
