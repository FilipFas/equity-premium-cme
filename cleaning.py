"""
cleaning.py
This module implements an object‐oriented preprocessing pipeline for multiple time‐series datasets. It performs:
    1) Reading and cleaning raw CSV data, including date filtering, missing value imputation, and equity premium computation.
    2) Log transformation and standardization of specified variables.
    3) Feature selection, target renaming, and lagged feature construction.
    4) Rolling‐window split generation for out‐of‐sample validation across annual, quarterly, and monthly frequencies.
Author: Filippo Fasoli
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import FEATURE_SELECTION, ROLLING_SPLIT_PARAMS


class DataCleaner:
    """
    DataCleaner handles the preprocessing pipeline for a single frequency dataset.

    This class reads raw CSV data, applies several preprocessing steps such as 
    missing value imputation, log transformation, and standardization. It also
    supports renaming the response variable and generating lagged features.

    Parameters
    ----------
    filepath : str
        Path to the input CSV file.
    frequency : str
        One of {"annual", "quarterly", "monthly"}, used for naming and config access.
    output_dir : str
        Directory where output CSVs will be saved.
    log_vars : list of str, optional
        Names of variables to apply log transformation.
    std_vars : list of str, optional
        Names of variables to be standardized.
    date_col : str, optional
        Column name to use for date-based filtering.
    start_date : str, optional
        Earliest date to include in the dataset.
    drop_threshold : float, optional
        Maximum allowed fraction of missing values for a column to be retained.
    """
    
    def __init__(self, filepath, frequency, output_dir, log_vars=None, std_vars=None,
                 date_col=None, start_date=None, drop_threshold=0.5):
        self.filepath = filepath
        self.frequency = frequency
        self.output_dir = output_dir
        self.log_vars = log_vars or []
        self.std_vars = std_vars or []
        self.date_col = date_col
        self.start_date = start_date
        self.drop_threshold = drop_threshold
        self.scaler = StandardScaler()
        self.df = pd.read_csv(filepath)
        self.dynamic_df = None
        os.makedirs(output_dir, exist_ok=True)

    def clean(self):
        """
        Clean the raw dataset:
        
        - Filter by date if specified
        - Drop columns with excessive missing values
        - Impute missing values using forward/backward fill
        - Compute equity premium if 'ret' and 'Rfree' are present
        - Save the cleaned DataFrame to disk
        """
        if self.date_col and self.start_date:
            self.df = self.df[self.df[self.date_col] >= self.start_date]

        to_drop = self.df.columns[self.df.isnull().mean() > self.drop_threshold]
        self.df.drop(columns=to_drop, inplace=True)

        self.df.fillna(method='ffill', inplace=True)
        self.df.fillna(method='bfill', inplace=True)

        if 'ret' in self.df.columns and 'Rfree' in self.df.columns:
            self.df[f'equity_premium_{self.frequency}'] = self.df['ret'] - self.df['Rfree']

        out_path = os.path.join(self.output_dir, f'data_{self.frequency}_cleaned.csv')
        self.df.to_csv(out_path, index=False)

    def apply_log_transforms(self):
        """
        Apply log transformation to variables listed in self.log_vars.

        Each transformed variable is saved as a new column with suffix '_log'.
        Output is saved to disk.
        """
        for var in self.log_vars:
            if var in self.df.columns:
                self.df[f'{var}_log'] = np.log(self.df[var])

        out_path = os.path.join(self.output_dir, f'data_{self.frequency}_log_transformed.csv')
        self.df.to_csv(out_path, index=False)

    def standardize(self):
        """
        Standardize variables listed in self.std_vars to have zero mean and unit variance.

        Variables not found in the DataFrame are ignored. The transformed dataset
        is saved to disk.
        """
        df_copy = self.df.copy()
        present_vars = [v for v in self.std_vars if v in df_copy.columns]
        df_copy[present_vars] = self.scaler.fit_transform(df_copy[present_vars])
        self.df = df_copy

        out_path = os.path.join(self.output_dir, f'data_{self.frequency}_standardized.csv')
        self.df.to_csv(out_path, index=False)

    def select_features(self, keep_cols, drop_cols=None):
        """
        Retain only selected features and rename target variable to 'y'.

        Parameters
        ----------
        keep_cols : list of str
            Columns to retain in the dataset.
        drop_cols : list of str, optional
            Columns to drop before filtering with keep_cols.
        """
        self.df = self.df.rename(columns={f'equity_premium_{self.frequency}': 'y'})

        if drop_cols:
            self.df.drop(columns=drop_cols, inplace=True, errors='ignore')

        self.df = self.df[keep_cols]

    @staticmethod
    def make_lags(df, p=4, q=1):
        """
        Create lagged features for predictors and target.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with predictors and target ('y') column.
        p : int
            Number of lags for predictors.
        q : int
            Number of lags for target.

        Returns
        -------
        pandas.DataFrame
            DataFrame including lagged predictors and target, with NaNs dropped.
        """
        out = df.copy()
        predictors = list(out.columns.drop('y'))

        for col in predictors:
            for lag in range(1, p + 1):
                out[f"{col}_lag{lag}"] = out[col].shift(lag)

        for lag in range(1, q + 1):
            out[f"y_lag{lag}"] = out['y'].shift(lag)

        return out.dropna().reset_index(drop=True)

    def get_dynamic_df(self):
        """
        Get the internal DataFrame with lagged features.

        Returns
        -------
        pandas.DataFrame
            DataFrame with lagged features.
        """
        return self.dynamic_df

    def summary(self):
        """
        Print non-null count of each variable in the dataset.
        Useful for quick diagnostics of missing data or preprocessing errors.
        """
        print(f"\n{self.frequency.capitalize()} dataset - Non-null count:")
        print(self.df.count())


class RollingWindowSplitter:
    """
    Generator of rolling window splits for time-series validation.

    Parameters
    ----------
    data : pandas.DataFrame
        Time-ordered dataset with lagged features.
    train_size : int
        Number of samples in the training window.
    val_size : int
        Number of samples in the validation window.
    test_size : int
        Number of samples in the test window.
    step_size : int
        Step size for rolling the window forward.
    """

    def __init__(self, data, train_size, val_size, test_size, step_size):
        self.data = data
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.step_size = step_size

    def generate_splits(self):
        """
        Yield rolling (train, validation, test) splits.

        Returns
        -------
        generator of tuple
            Each tuple is (train_df, val_df, test_df) for a rolling window.
        """
        start = 0
        total = self.train_size + self.val_size + self.test_size

        while start + total <= len(self.data):
            train_end = start + self.train_size
            val_end = train_end + self.val_size
            test_end = val_end + self.test_size

            yield (
                self.data.iloc[start:train_end],
                self.data.iloc[train_end:val_end],
                self.data.iloc[val_end:test_end]
            )

            start += self.step_size


class MultiFrequencyCleaner:
    """
    Orchestrates end-to-end preprocessing for multiple time frequencies.

    Combines several DataCleaner instances and applies their full pipelines.
    Stores and returns rolling splits for downstream use.

    Parameters
    ----------
    config : dict
        Dictionary specifying settings for each frequency, including:
        - filepath, output_dir, log_vars, std_vars
        - date_col, start_date
    """

    def __init__(self, config):
        self.cleaners = []
        self.splits = {}

        for freq, cfg in config.items():
            cleaner = DataCleaner(
                filepath=cfg['filepath'],
                frequency=freq,
                output_dir=cfg['output_dir'],
                log_vars=cfg.get('log_vars', []),
                std_vars=cfg.get('std_vars', []),
                date_col=cfg['date_col'],
                start_date=cfg['start_date']
            )
            self.cleaners.append(cleaner)

    def run_all(self):
        """
        Execute the full pipeline for all configured datasets:
        - Cleaning, transformation, feature selection
        - Lag generation and rolling window splitting
        - Store all splits by frequency
        """
        for cleaner in self.cleaners:
            cleaner.clean()
            cleaner.apply_log_transforms()
            cleaner.standardize()
            cleaner.summary()

            fs = FEATURE_SELECTION[cleaner.frequency]
            cleaner.select_features(keep_cols=fs["keep"], drop_cols=fs["drop"])

            cleaner.dynamic_df = DataCleaner.make_lags(cleaner.df, p=4, q=1)

            split_params = ROLLING_SPLIT_PARAMS[cleaner.frequency]
            splitter = RollingWindowSplitter(cleaner.get_dynamic_df(), **split_params)

            splits = list(splitter.generate_splits())
            self.splits[cleaner.frequency] = splits
            print(f"Created {len(splits)} rolling splits for {cleaner.frequency}")

    def get_splits(self):
        """
        Return dictionary of rolling window splits for each frequency.

        Returns
        -------
        dict
            Keys are frequency strings, values are lists of (train, val, test) tuples.
        """
        return self.splits


#######################
#### USAGE EXAMPLE ####
#######################


# from cleaning import MultiFrequencyCleaner

# config = {
#     'annual': {
#         'filepath': '/Users/FilippoFasoli_1/Desktop/THESIS/New_Data/Data2023_annual.csv',
#         'output_dir': '/Users/FilippoFasoli_1/Desktop/THESIS/New_Data/',
#         'log_vars': ['cfacc', 'eqis'],
#         'std_vars': ['accrul', 'gpce', 'cfacc_log', 'eqis_log'],
#         'date_col': 'yyyy',
#         'start_date': 1965
#     },
#     'quarterly': {
#         'filepath': '/Users/FilippoFasoli_1/Desktop/THESIS/New_Data/Data2023_quarterly.csv',
#         'output_dir': '/Users/FilippoFasoli_1/Desktop/THESIS/New_Data/',
#         'std_vars': ['pce', 'crdstd', 'i/k'],
#         'date_col': 'yyyyq',
#         'start_date': 19902
#     },
#     'monthly': {
#         'filepath': '/Users/FilippoFasoli_1/Desktop/THESIS/New_Data/Data2023_monthly.csv',
#         'output_dir': '/Users/FilippoFasoli_1/Desktop/THESIS/New_Data/',
#         'log_vars': ['tbl', 'lty'],
#         'std_vars': ['tbl_log', 'lty_log'],
#         'date_col': 'yyyymm',
#         'start_date': 192601
#     }
# }

# cleaner = MultiFrequencyCleaner(config)
# cleaner.run_all()

# # Access the rolling splits
# splits = cleaner.get_splits()
# splits['annual'][0]  # first (train, val, test) tuple for annual
