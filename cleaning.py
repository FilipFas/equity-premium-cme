# Full code for cleaning.py with OOP structure

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataCleaner:
    """
    DataCleaner handles the preprocessing pipeline for a single frequency dataset.
    It reads raw data, cleans missing values, applies transformations, and prepares
    the dataset for feature selection and lag creation.
    """

    def __init__(self, filepath, frequency, output_dir, log_vars=None, std_vars=None,
                 date_col=None, start_date=None, drop_threshold=0.5):
        # Initialize file paths, transformation parameters, and output settings
        self.filepath = filepath
        self.frequency = frequency
        self.output_dir = output_dir
        self.log_vars = log_vars or []         # Variables to log-transform
        self.std_vars = std_vars or []         # Variables to standardize
        self.date_col = date_col               # Column name for date filtering
        self.start_date = start_date           # Earliest date to include
        self.drop_threshold = drop_threshold   # Max allowed fraction of missing values
        self.scaler = StandardScaler()         # Standard scaler for standardization

        # Load data and prepare output directory
        self.df = pd.read_csv(filepath)
        self.dynamic_df = None
        os.makedirs(output_dir, exist_ok=True)

    def clean(self):
        """
        Perform initial cleaning:
        - Filter rows by date
        - Drop columns with too many missing values
        - Forward/backward fill remaining NaNs
        - Compute equity premium if returns are available
        - Save cleaned data to CSV
        """
        if self.date_col and self.start_date:
            self.df = self.df[self.df[self.date_col] >= self.start_date]

        # Drop columns with a higher missing rate than threshold
        to_drop = self.df.columns[self.df.isnull().mean() > self.drop_threshold]
        self.df.drop(columns=to_drop, inplace=True)

        # Impute remaining missing values
        self.df.fillna(method='ffill', inplace=True)
        self.df.fillna(method='bfill', inplace=True)

        # Compute equity premium: ret - Rfree
        if 'ret' in self.df.columns and 'Rfree' in self.df.columns:
            self.df[f'equity_premium_{self.frequency}'] = self.df['ret'] - self.df['Rfree']

        # Export cleaned dataset
        out_path = os.path.join(self.output_dir, f'data_{self.frequency}_cleaned.csv')
        self.df.to_csv(out_path, index=False)

    def apply_log_transforms(self):
        """
        Apply natural log transformation to specified variables and save result.
        """
        for var in self.log_vars:
            if var in self.df.columns:
                self.df[f'{var}_log'] = np.log(self.df[var])

        out_path = os.path.join(self.output_dir, f'data_{self.frequency}_log_transformed.csv')
        self.df.to_csv(out_path, index=False)

    def standardize(self):
        """
        Standardize specified variables to zero mean and unit variance.
        Saves standardized dataset to CSV.
        """
        df_copy = self.df.copy()
        present_vars = [v for v in self.std_vars if v in df_copy.columns]
        df_copy[present_vars] = self.scaler.fit_transform(df_copy[present_vars])
        self.df = df_copy

        out_path = os.path.join(self.output_dir, f'data_{self.frequency}_standardized.csv')
        self.df.to_csv(out_path, index=False)

    def select_features(self, keep_cols, drop_cols=None):
        """
        Narrow dataset to chosen predictors:
        - Rename equity premium column to 'y'
        - Drop irrelevant columns
        - Keep only specified columns
        """
        # Rename target column
        self.df = self.df.rename(columns={f'equity_premium_{self.frequency}': 'y'})

        # Drop unwanted columns
        if drop_cols:
            self.df.drop(columns=drop_cols, inplace=True, errors='ignore')

        # Retain only selected features
        self.df = self.df[keep_cols]
    @staticmethod
    def make_lags(df, p=4, q=1):
        """
        Static method to create lagged predictors and target from a given DataFrame.
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
        Return the dataset with lagged features for rolling window splitting.
        """
        return self.dynamic_df

    def summary(self):
        """
        Print non-null counts for each column in the current DataFrame.
        """
        print(f"\n{self.frequency.capitalize()} dataset - Non-null count:")
        print(self.df.count())


class RollingWindowSplitter:
    """
    RollingWindowSplitter generates sequential train/validation/test splits
    from a time-ordered dataset using fixed window sizes and step increments.
    """

    def __init__(self, data, train_size, val_size, test_size, step_size):
        self.data = data                # Time-ordered DataFrame
        self.train_size = train_size    # Number of observations in training set
        self.val_size = val_size        # Number of observations in validation set
        self.test_size = test_size      # Number of observations in test set
        self.step_size = step_size      # Step increment for rolling forward

    def generate_splits(self):
        """
        Yield tuples of (train_df, val_df, test_df) for each rolling window.
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
    MultiFrequencyCleaner orchestrates preprocessing for multiple datasets
    at different frequencies and generates rolling splits for each.
    """

    def __init__(self, config):
        self.cleaners = []  # List of DataCleaner instances
        self.splits = {}    # Dictionary to hold splits by frequency

        # Instantiate a DataCleaner for each frequency in the config
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
        Execute the full pipeline for each DataCleaner:
        cleaning, transformations, feature selection, lag creation,
        and rolling window splitting.
        """
        for cleaner in self.cleaners:
            # Core preprocessing steps
            cleaner.clean()
            cleaner.apply_log_transforms()
            cleaner.standardize()
            cleaner.summary()

            # Frequency-specific feature selection and split parameters
            if cleaner.frequency == 'annual':
                cleaner.select_features(
                    keep_cols=["y", "eqis", "ltr", "corpr", "infl", "cay",
                               "fbm", "dtoy", "rdsp", "gip", "tchi"],
                    drop_cols=["yyyy", "Rfree", "ret"]
                )
                cleaner.make_lags(p=4, q=1)
                splitter = RollingWindowSplitter(cleaner.get_dynamic_df(),
                                                 train_size=30, val_size=10,
                                                 test_size=10, step_size=1)

            elif cleaner.frequency == 'quarterly':
                cleaner.select_features(
                    keep_cols=["y", "pce", "crdstd", "i/k", "shtint",
                               "b/m", "vp", "dtoy"],
                    drop_cols=["yyyyq", "Rfree", "ret"]
                )
                cleaner.make_lags(p=4, q=1)
                splitter = RollingWindowSplitter(cleaner.get_dynamic_df(),
                                                 train_size=80, val_size=30,
                                                 test_size=10, step_size=5)

            elif cleaner.frequency == 'monthly':
                cleaner.select_features(
                    keep_cols=["y", "ygap", "tbl", "skvw", "tail",
                               "dtoy", "dtoat", "lty"],
                    drop_cols=["yyyymm", "Rfree", "ret"]
                )
                cleaner.make_lags(p=4, q=1)
                splitter = RollingWindowSplitter(cleaner.get_dynamic_df(),
                                                 train_size=800, val_size=200,
                                                 test_size=100, step_size=25)

            # Generate and store rolling splits
            splits = list(splitter.generate_splits())
            self.splits[cleaner.frequency] = splits
            print(f"Created {len(splits)} rolling splits for {cleaner.frequency}")

    def get_splits(self):
        """
        Return the generated rolling window splits for each frequency.
        """
        return self.splits


#######################
#### USAGE EXAMPLE ####
#######################


# from cleaning import MultiFrequencyCleaner

# config = {
#     'annual': {
#         'filepath': '/Your_Path/Data2023_annual.csv',
#         'output_dir': '/Your_Path',
#         'log_vars': ['cfacc', 'eqis'],
#         'std_vars': ['accrul', 'gpce', 'cfacc_log', 'eqis_log'],
#         'date_col': 'yyyy',
#         'start_date': 1965
#     },
#     'quarterly': {
#         'filepath': '/Your_Path/Data2023_quarterly.csv',
#         'output_dir': '/Your_Path',
#         'std_vars': ['pce', 'crdstd', 'i/k'],
#         'date_col': 'yyyyq',
#         'start_date': 19902
#     },
#     'monthly': {
#         'filepath': '/UYour_Path/Data2023_monthly.csv',
#         'output_dir': '/Your_Path',
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
