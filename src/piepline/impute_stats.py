import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from impute_ml_dl import missforest_imputation, mlp_imputation, autoencoder_imputation, gain_imputation
import logging

logger = logging.getLogger(__name__)

def mean_imputation(data, col_miss=['X1', 'X2']):
    """
    Impute missing values with column means.
    """
    dat_imputed = data.copy()
    for col in col_miss:
        if dat_imputed[col].isna().all():
            logger.warning(f"All values in {col} are NaN, filling with 0")
            dat_imputed[col] = dat_imputed[col].fillna(0)
        else:
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
        X_train = dat_imputed.loc[mask, predictors]
        y_train = data.loc[mask, col]
        # Check for NaNs in X_train
        if X_train.isna().any().any():
            logger.warning(f"NaN values in predictors for {col} in single imputation, filling with mean")
            X_train = X_train.fillna(X_train.mean())
        model = LinearRegression().fit(X_train, y_train)
        mask_missing = data[col].isna()
        X_missing = dat_imputed.loc[mask_missing, predictors]
        if len(X_missing) > 0:
            if X_missing.isna().any().any():
                logger.warning(f"NaN values in X_missing for {col} in single imputation, filling with mean")
                X_missing = X_missing.fillna(X_missing.mean())
            dat_imputed.loc[mask_missing, col] = model.predict(X_missing)
    for out in ['y', 'y_score']:
        if out in original_data.columns and out not in dat_imputed.columns:
            dat_imputed[out] = original_data[out]
    return [dat_imputed]

def mice_imputation(data, original_data, outcome=None, col_miss=['X1', 'X2'], seed=123):
    """
    Impute missing values using MICE with linear regression.
    """
    np.random.seed(seed)
    dat_imputed_list = []
    predictors = [col for col in data.columns if col not in ['y', 'y_score']]
    if outcome:
        predictors.append(outcome)
    for i in range(2):  # Two imputations
        logger.info(f"Starting MICE imputation {i+1}/2")
        imputer = IterativeImputer(
            estimator=LinearRegression(),
            max_iter=10,
            tol=0.001,
            random_state=seed + i
        )
        dat_imputed = data.copy()
        dat_imputed[col_miss] = imputer.fit_transform(dat_imputed[predictors])[..., [predictors.index(col) for col in col_miss]]
        for out in ['y', 'y_score']:
            if out in original_data.columns and out not in dat_imputed.columns:
                dat_imputed[out] = original_data[out]
        dat_imputed_list.append(dat_imputed)
    return dat_imputed_list

def complete_data(data, original_data, outcome=None, col_miss=['X1', 'X2']):
    """
    Return complete-case dataset by removing rows with missing values in col_miss.
    """
    dat_complete = data.dropna(subset=col_miss)
    if len(dat_complete) == 0:
        logger.error(f"No complete cases remain after dropping rows with missing values in {col_miss}")
        # Fallback to original_data with mean imputation
        dat_complete = original_data.copy()
        for col in col_miss:
            if dat_complete[col].isna().all():
                logger.warning(f"All values in {col} are NaN in complete_data fallback, filling with 0")
                dat_complete[col] = dat_complete[col].fillna(0)
            else:
                dat_complete[col] = dat_complete[col].fillna(dat_complete[col].mean())
    elif len(dat_complete) < 20:
        logger.warning(f"Too few complete cases ({len(dat_complete)}) after dropping rows with missing values in {col_miss}")
    logger.info(f"Complete-case dataset has {len(dat_complete)} rows")
    return [dat_complete]

def impute_datasets(data, complete, col_miss=['X1', 'X2'], seed=123):
    """
    Apply imputation methods to dataset with missing values.
    """
    imputed_datasets = {
        'complete_data': {'full': complete_data(data, complete, outcome=None, col_miss=col_miss)},
        'mean': {'full': mean_imputation(data, col_miss=col_miss)},
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
- mice_imputation: Imputes X1, X2 with MICE (2 imputations)
- complete_data: Returns complete-case dataset by removing rows with missing values in X1, X2
- impute_datasets: Applies 19 imputation methods + complete-case baseline
"""