import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from impute_ml_dl import missforest_imputation, mlp_imputation, autoencoder_imputation, gain_imputation

def mean_imputation(data, col_miss=['X1', 'X2']):
    """
    Impute missing values with column means.
    """
    dat_imputed = data.copy()
    for col in col_miss:
        dat_imputed[col] = dat_imputed[col].fillna(dat_imputed[col].mean())
    return [dat_imputed]

def single_imputation(data, original_data, outcome=None, col_miss=['X1', 'X2']):
    """
    Impute missing values using linear regression.
    """
    dat_imputed = data.copy()
    predictors = [col for col in data.columns if col not in col_miss + ['y', 'y_score']]
    if outcome:
        predictors.append(outcome)
    for col in col_miss:
        mask = ~data[col].isna()
        X_train = dat_imputed.loc[mask, predictors]  # Use current fills
        y_train = data.loc[mask, col]
        model = LinearRegression().fit(X_train, y_train)
        mask_missing = data[col].isna()
        X_missing = dat_imputed.loc[mask_missing, predictors]
        if len(X_missing) > 0:
            dat_imputed.loc[mask_missing, col] = model.predict(X_missing)
    for out in ['y', 'y_score']:
        if out in original_data.columns and out not in dat_imputed.columns:
            dat_imputed[out] = original_data[out]
    return [dat_imputed]

def mice_imputation(data, original_data, outcome=None, col_miss=['X1', 'X2'], n_imputations=5, seed=123):
    """
    Impute missing values using MICE.
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

def complete_data(data, col_miss=['X1', 'X2']):
    """
    Return complete data without imputation (baseline).
    """
    return [data.copy()]

def impute_datasets(datasets, col_miss=['X1', 'X2'], seed=123):
    """
    Apply imputation methods to datasets, including complete-data baseline.
    """
    imputed_datasets = {}
    for dataset_name, data in datasets.items():
        # Initialize with complete data
        complete = data.copy()
        if dataset_name == 'complete_data':
            data = complete
        else:
            data = data.copy()
        
        imputed_datasets[dataset_name] = {
            'complete_data': {'full': complete_data(data, col_miss)},
            'mean': {'full': mean_imputation(data, col_miss)},
            'single_with_y': {'full': single_imputation(data, complete, outcome='y', col_miss=col_miss)},
            'single_with_y_score': {'full': single_imputation(data, complete, outcome='y_score', col_miss=col_miss)},
            'single_without': {'full': single_imputation(data.drop(columns=['y', 'y_score'], errors='ignore'), complete, outcome=None, col_miss=col_miss)},
            'mice_with_y': {'full': mice_imputation(data, complete, outcome='y', col_miss=col_miss, seed=seed)},
            'mice_with_y_score': {'full': mice_imputation(data, complete, outcome='y_score', col_miss=col_miss, seed=seed)},
            'mice_without': {'full': mice_imputation(data.drop(columns=['y', 'y_score'], errors='ignore'), complete, outcome=None, col_miss=col_miss, seed=seed)},
            'missforest_with_y': {'full': missforest_imputation(data, complete, outcome='y', col_miss=col_miss, seed=seed)},
            'missforest_with_y_score': {'full': missforest_imputation(data, complete, outcome='y_score', col_miss=col_miss, seed=seed)},
            'missforest_without': {'full': missforest_imputation(data.drop(columns=['y', 'y_score'], errors='ignore'), complete, outcome=None, col_miss=col_miss, seed=seed)},
            'mlp_with_y': {'full': mlp_imputation(data, complete, outcome='y', col_miss=col_miss, seed=seed)},
            'mlp_with_y_score': {'full': mlp_imputation(data, complete, outcome='y_score', col_miss=col_miss, seed=seed)},
            'mlp_without': {'full': mlp_imputation(data.drop(columns=['y', 'y_score'], errors='ignore'), complete, outcome=None, col_miss=col_miss, seed=seed)},
            'ae_with_y': {'full': autoencoder_imputation(data, complete, outcome='y', col_miss=col_miss, seed=seed)},
            'ae_with_y_score': {'full': autoencoder_imputation(data, complete, outcome='y_score', col_miss=col_miss, seed=seed)},
            'ae_without': {'full': autoencoder_imputation(data.drop(columns=['y', 'y_score'], errors='ignore'), complete, outcome=None, col_miss=col_miss, seed=seed)},
            'gain_with_y': {'full': gain_imputation(data, complete, outcome='y', col_miss=col_miss, seed=seed)},
            'gain_with_y_score': {'full': gain_imputation(data, complete, outcome='y_score', col_miss=col_miss, seed=seed)},
            'gain_without': {'full': gain_imputation(data.drop(columns=['y', 'y_score'], errors='ignore'), complete, outcome=None, col_miss=col_miss, seed=seed)}
        }
    return imputed_datasets

# Documentation
"""
Functions:
- mean_imputation: Imputes X1, X2 with means
- single_imputation: Imputes X1, X2 with linear regression
- mice_imputation: Imputes X1, X2 with MICE (5 imputations)
- complete_data: Returns complete data unchanged (baseline)
- impute_datasets: Applies 19 imputation methods + complete-data baseline
"""