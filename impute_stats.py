import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression

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

def single_imputation(data, outcome='y', col_miss=['X1', 'X2']):
    """
    Impute missing values using linear regression.
    
    Parameters:
    - data: Input DataFrame
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
    return [dat_imputed]

def mice_imputation(data, outcome='y', col_miss=['X1', 'X2'], n_imputations=5, seed=123):
    """
    Impute missing values using MICE.
    
    Parameters:
    - data: Input DataFrame
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
        imputed_datasets[dataset_name] = {}
        for outcome in ['y', 'y_score']:
            imputed_datasets[dataset_name][f'mean_{outcome}'] = {
                'full': mean_imputation(data, col_miss)
            }
            imputed_datasets[dataset_name][f'single_with_{outcome}'] = {
                'full': single_imputation(data, outcome=outcome, col_miss=col_miss)
            }
            imputed_datasets[dataset_name][f'single_without_{outcome}'] = {
                'full': single_imputation(data.drop(columns=[outcome]), outcome=None, col_miss=col_miss)
            }
            imputed_datasets[dataset_name][f'mice_with_{outcome}'] = {
                'full': mice_imputation(data, outcome=outcome, col_miss=col_miss, seed=seed)
            }
            imputed_datasets[dataset_name][f'mice_without_{outcome}'] = {
                'full': mice_imputation(data.drop(columns=[outcome]), outcome=None, col_miss=col_miss, seed=seed)
            }
    return imputed_datasets

# Documentation
"""
Functions:
- mean_imputation: Imputes X1, X2 with means
- single_imputation: Imputes X1, X2 with linear regression
- mice_imputation: Imputes X1, X2 with MICE (5 imputations)
- impute_datasets: Applies all imputation methods
"""