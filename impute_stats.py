import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression

# Import ML/DL methods
from impute_ml_dl import missforest_imputation, mlp_imputation, autoencoder_imputation, gain_imputation

def mean_imputation(data, col_miss=['X1', 'X2']):
    """
    Impute missing values with column means.
    
    Parameters:
    - data: Input DataFrame
    - col_miss: Columns to impute
    
    Returns:
    - List of one imputed DataFrame
    """
    dat_imputed = data.copy()
    for col in col_miss:
        dat_imputed[col] = dat_imputed[col].fillna(dat_imputed[col].mean())
    return [dat_imputed]

def single_imputation(data, original_data, outcome=None, col_miss=['X1', 'X2']):
    """
    Impute missing values using linear regression.
    
    Parameters:
    - data: Input DataFrame (possibly without outcomes)
    - original_data: Original DataFrame with outcomes for restoration
    - outcome: Response variable to include (or None)
    - col_miss: Columns to impute
    
    Returns:
    - List of one imputed DataFrame
    """
    dat_imputed = data.copy()
    predictors = [col for col in data.columns if col not in col_miss + ['y', 'y_score']]
    if outcome:
        predictors.append(outcome)
    for col in col_miss:
        mask = ~data[col].isna()
        X_train = data.loc[mask, predictors]
        y_train = data.loc[mask, col]
        model = LinearRegression().fit(X_train, y_train)
        mask_missing = data[col].isna()
        X_missing = data.loc[mask_missing, predictors]
        if len(X_missing) > 0:
            dat_imputed.loc[mask_missing, col] = model.predict(X_missing)
    for out in ['y', 'y_score']:
        if out in original_data.columns and out not in dat_imputed.columns:
            dat_imputed[out] = original_data[out]
    return [dat_imputed]

def mice_imputation(data, original_data, outcome=None, col_miss=['X1', 'X2'], n_imputations=5, seed=123):
    """
    Impute missing values using MICE.
    
    Parameters:
    - data: Input DataFrame (possibly without outcomes)
    - original_data: Original DataFrame with outcomes for restoration
    - outcome: Response variable to include (or None)
    - col_miss: Columns to impute
    - n_imputations: Number of imputations
    - seed: Random seed
    
    Returns:
    - List of imputed DataFrames
    """
    np.random.seed(seed)
    dat_imputed_list = []
    predictors = [col for col in data.columns if col not in ['y', 'y_score']]
    if outcome:
        predictors.append(outcome)
    for i in range(n_imputations):
        imputer = IterativeImputer(max_iter=10, random_state=seed + i)
        dat_imputed = data.copy()
        dat_imputed[predictors] = imputer.fit_transform(data[predictors])
        for out in ['y', 'y_score']:
            if out in original_data.columns and out not in dat_imputed.columns:
                dat_imputed[out] = original_data[out]
        dat_imputed_list.append(dat_imputed)
    return dat_imputed_list

def impute_datasets(datasets, col_miss=['X1', 'X2'], seed=123):
    """
    Apply imputation methods to datasets.
    
    Parameters:
    - datasets: Dictionary of datasets
    - col_miss: Columns to impute
    - seed: Random seed
    
    Returns:
    - Dictionary of imputed datasets
    """
    imputed_datasets = {}
    for dataset_name, data in datasets.items():
        imputed_datasets[dataset_name] = {
            'mean': {'full': mean_imputation(data, col_miss)},
            'single_with_y': {'full': single_imputation(data, data, outcome='y', col_miss=col_miss)},
            'single_with_y_score': {'full': single_imputation(data, data, outcome='y_score', col_miss=col_miss)},
            'single_without': {'full': single_imputation(data.drop(columns=['y', 'y_score'], errors='ignore'), data, outcome=None, col_miss=col_miss)},
            'mice_with_y': {'full': mice_imputation(data, data, outcome='y', col_miss=col_miss, seed=seed)},
            'mice_with_y_score': {'full': mice_imputation(data, data, outcome='y_score', col_miss=col_miss, seed=seed)},
            'mice_without': {'full': mice_imputation(data.drop(columns=['y', 'y_score'], errors='ignore'), data, outcome=None, col_miss=col_miss, seed=seed)},
            'missforest_with_y': {'full': missforest_imputation(data, data, outcome='y', col_miss=col_miss, seed=seed)},
            'missforest_with_y_score': {'full': missforest_imputation(data, data, outcome='y_score', col_miss=col_miss, seed=seed)},
            'missforest_without': {'full': missforest_imputation(data.drop(columns=['y', 'y_score'], errors='ignore'), data, outcome=None, col_miss=col_miss, seed=seed)},
            'mlp_with_y': {'full': mlp_imputation(data, data, outcome='y', col_miss=col_miss, seed=seed)},
            'mlp_with_y_score': {'full': mlp_imputation(data, data, outcome='y_score', col_miss=col_miss, seed=seed)},
            'mlp_without': {'full': mlp_imputation(data.drop(columns=['y', 'y_score'], errors='ignore'), data, outcome=None, col_miss=col_miss, seed=seed)},
            'ae_with_y': {'full': autoencoder_imputation(data, data, outcome='y', col_miss=col_miss, seed=seed)},
            'ae_with_y_score': {'full': autoencoder_imputation(data, data, outcome='y_score', col_miss=col_miss, seed=seed)},
            'ae_without': {'full': autoencoder_imputation(data.drop(columns=['y', 'y_score'], errors='ignore'), data, outcome=None, col_miss=col_miss, seed=seed)},
            'gain_with_y': {'full': gain_imputation(data, data, outcome='y', col_miss=col_miss, seed=seed)},
            'gain_with_y_score': {'full': gain_imputation(data, data, outcome='y_score', col_miss=col_miss, seed=seed)},
            'gain_without': {'full': gain_imputation(data.drop(columns=['y', 'y_score'], errors='ignore'), data, outcome=None, col_miss=col_miss, seed=seed)}
        }
    return imputed_datasets

# Documentation
"""
Functions:
- mean_imputation: Imputes X1, X2 with means
- single_imputation: Imputes X1, X2 with linear regression
- mice_imputation: Imputes X1, X2 with MICE (5 imputations)
- impute_datasets: Applies 19 imputation methods (mean, single_with_y, single_with_y_score, single_without, mice_with_y, mice_with_y_score, mice_without, missforest_with_y, missforest_with_y_score, missforest_without, mlp_with_y, mlp_with_y_score, mlp_without, ae_with_y, ae_with_y_score, ae_without, gain_with_y, gain_with_y_score, gain_without)
"""