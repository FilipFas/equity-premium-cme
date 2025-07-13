import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.metrics.pairwise import laplacian_kernel, polynomial_kernel


class ConditionalMeanEmbedding:
    """
    Object-oriented class to handle Conditional Mean Embedding (CME) operations.

    Features:
    - Kernel-based Gram matrix computation
    - Fitting of CME operator (regularized)
    - Prediction on new data
    - Pooled R^2 evaluation over multiple train-validation splits
    """

    def __init__(self, kernel_type, length_scale=None, lam=1e-3,
                 poly_degree=None, poly_gamma=None, poly_coef0=None):
        """
        Initialize the CME model with kernel and regularization parameters.

        Parameters:
        - kernel_type: str, one of {'rbf', 'ard-rbf', 'laplacian', 'matern', 'polynomial'}
        - length_scale: float or array-like, kernel length scale(s)
        - lam: float, regularization parameter
        - poly_degree: int, degree for polynomial kernel (only for 'polynomial')
        - poly_gamma: float, gamma for polynomial kernel
        - poly_coef0: float, coef0 for polynomial kernel
        """
        self.kernel_type = kernel_type
        self.length_scale = length_scale
        self.lam = lam
        self.poly_degree = poly_degree
        self.poly_gamma = poly_gamma
        self.poly_coef0 = poly_coef0

        self.X_train = None  # To store training data
        self.beta = None     # To store CME operator application to targets

    def _compute_gram(self, X, Y):
        """
        Internal method to compute the kernel Gram matrix between X and Y.

        Returns:
        - Gram matrix of shape (len(X), len(Y))
        """
        kt = self.kernel_type
        ls = self.length_scale
        pdg, pg, p0 = self.poly_degree, self.poly_gamma, self.poly_coef0
        d = X.shape[1]

        if kt == "rbf":
            return RBF(length_scale=ls)(X, Y)
        if kt == "ard-rbf":
            arr = np.atleast_1d(ls).ravel()
            if arr.size == 0:
                ls_vec = np.ones(d)
            elif arr.size == d - 1:
                ls_vec = np.concatenate([arr, [arr[-1]]])
            elif arr.size == d:
                ls_vec = arr
            elif arr.size == 1:
                ls_vec = np.ones(d) * arr.item()
            else:
                raise ValueError(f"Invalid ARD shape: {arr.size}")
            return RBF(length_scale=ls_vec)(X, Y)
        if kt == "laplacian":
            return laplacian_kernel(X, Y, gamma=1.0 / ls)
        if kt == "matern":
            return Matern(length_scale=ls, nu=2.5)(X, Y)
        if kt == "polynomial":
            return polynomial_kernel(X, Y, degree=pdg, gamma=pg, coef0=p0)

        raise ValueError(f"Unsupported kernel type: {kt}")

    def fit(self, X, y):
        """
        Fit the CME operator on the training data X, y.

        Stores:
        - X as self.X_train
        - CME coefficients beta as self.beta
        """
        self.X_train = X
        m = X.shape[0]
        K = self._compute_gram(X, X)
        H = np.eye(m) - np.ones((m, m)) / m
        C = np.linalg.solve(H @ K + self.lam * np.eye(m), H @ K)
        self.beta = C @ y

    def predict(self, X_test):
        """
        Predict output y for new input X_test using fitted CME operator.

        Returns:
        - y_pred: array of predictions
        """
        if self.X_train is None or self.beta is None:
            raise ValueError("CME model must be fit before prediction.")
        K_test = self._compute_gram(X_test, self.X_train)
        return K_test @ self.beta

    def pooled_r2(self, split_list):
        """
        Evaluate CME model across multiple train-validation splits.

        Parameters:
        - split_list: list of tuples (train_df, val_df, _) where:
            train_df and val_df must have features and a 'y' column

        Returns:
        - pooled R^2: float, defined as 1 - SSE_model / SSE_benchmark
        """
        all_y_true, all_y_pred, all_bench = [], [], []

        for train, val, _ in split_list:
            X_tr, y_tr = train.drop(columns="y").values, train["y"].values
            X_va, y_va = val.drop(columns="y").values, val["y"].values
            self.fit(X_tr, y_tr)
            y_pred = self.predict(X_va)

            # Rolling-mean benchmark
            history = list(y_tr)
            bench = [np.mean(history[:i+1]) for i in range(len(y_va))]

            all_y_true.extend(y_va)
            all_y_pred.extend(y_pred)
            all_bench.extend(bench)

        y = np.array(all_y_true)
        yhat = np.array(all_y_pred)
        bench = np.array(all_bench)

        sse_model = np.sum((y - yhat) ** 2)
        sse_bench = np.sum((y - bench) ** 2)
        return 1.0 - sse_model / sse_bench


def optuna_objective(trial, split_list, kernel_type):
    """
    Optuna-compatible objective function to optimize CME hyperparameters.

    Parameters:
    - trial: optuna.trial.Trial
    - split_list: list of (train, val, _) tuples
    - kernel_type: str, kernel to use for CME

    Returns:
    - Pooled R² score
    """
    if kernel_type == "ard-rbf":
        d = split_list[0][0].drop(columns="y").shape[1]
        ls = [trial.suggest_float(f"ard_ls_{i}", 1e-2, 1e1, log=True) for i in range(d)]
    else:
        ls = trial.suggest_float("length_scale", 1e-2, 1e1, log=True)

    lam = trial.suggest_float("lambda_reg", 1e-4, 1e1, log=True)
    pdg = pg = p0 = None
    if kernel_type == "polynomial":
        pdg = trial.suggest_int("poly_degree", 1, 3)
        pg = trial.suggest_float("poly_gamma", 1e-3, 1, log=True)
        p0 = trial.suggest_float("poly_coef0", 0.0, 1.0)

    model = ConditionalMeanEmbedding(
        kernel_type=kernel_type,
        length_scale=ls,
        lam=lam,
        poly_degree=pdg,
        poly_gamma=pg,
        poly_coef0=p0
    )
    return model.pooled_r2(split_list)
