"""Imputation method classes for simulation studies."""

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import logging
from tqdm import tqdm
from abc import ABC, abstractmethod
from numpy.random import default_rng

# Import GAN models (assume artifacts/models/gan_models.py exists as in original)
from artifacts.models.gan_models import Generator, Discriminator

logger = logging.getLogger(__name__)

class ImputationMethod(ABC):
    """Abstract base class for imputation methods.
    
    All imputation methods must implement:
    - impute(data, original_data, outcome=None, col_miss=['X1', 'X2'], n_imputations=1, seed=123): Return list of imputed DataFrames
    - name: Property for descriptive name
    """
    
    @abstractmethod
    def impute(self, data, original_data, outcome=None, col_miss=['X1', 'X2'], n_imputations=1, seed=123):
        pass
    
    @property
    @abstractmethod
    def name(self):
        pass

class CompleteData(ImputationMethod):
    def impute(self, data, original_data, outcome=None, col_miss=['X1', 'X2'], n_imputations=1, seed=123):
        dat_complete = data.dropna(subset=col_miss)
        if len(dat_complete) == 0:
            logger.error(f"No complete cases remain after dropping rows with missing values in {col_miss}")
            dat_complete = original_data.copy()
            for col in col_miss:
                if dat_complete[col].isna().all():
                    dat_complete[col] = dat_complete[col].fillna(0)
                else:
                    dat_complete[col] = dat_complete[col].fillna(dat_complete[col].mean())
        return [dat_complete]
    
    @property
    def name(self):
        return 'complete_data'

class MeanImputation(ImputationMethod):
    def impute(self, data, original_data, outcome=None, col_miss=['X1', 'X2'], n_imputations=1, seed=123):
        dat_imputed = data.copy()
        for col in col_miss:
            if dat_imputed[col].isna().all():
                logger.warning(f"All values in {col} are NaN, filling with 0")
                dat_imputed[col] = dat_imputed[col].fillna(0)
            else:
                dat_imputed[col] = dat_imputed[col].fillna(dat_imputed[col].mean())
        return [dat_imputed]
    
    @property
    def name(self):
        return 'mean'

# Add concrete classes for SingleImputation (with variants for with_y, with_y_score, without)
class SingleImputation(ImputationMethod):
    def __init__(self, use_outcome=None):
        self.use_outcome = use_outcome  # 'y', 'y_score', or None
    
    def impute(self, data, original_data, outcome=None, col_miss=['X1', 'X2'], n_imputations=1, seed=123):
        dat_imputed = data.copy()
        predictors = [col for col in data.columns if col not in col_miss + ['y', 'y_score']]
        if self.use_outcome:
            predictors.append(self.use_outcome)
        for col in col_miss:
            mask = ~data[col].isna()
            X_train = dat_imputed.loc[mask, predictors]
            y_train = data.loc[mask, col]
            if X_train.isna().any().any():
                X_train = X_train.fillna(X_train.mean())
            model = LinearRegression().fit(X_train, y_train)
            mask_missing = data[col].isna()
            X_missing = dat_imputed.loc[mask_missing, predictors]
            if len(X_missing) > 0:
                if X_missing.isna().any().any():
                    X_missing = X_missing.fillna(X_missing.mean())
                dat_imputed.loc[mask_missing, col] = model.predict(X_missing)
        for out in ['y', 'y_score']:
            if out in original_data.columns and out not in dat_imputed.columns:
                dat_imputed[out] = original_data[out]
        return [dat_imputed]
    
    @property
    def name(self):
        if self.use_outcome == 'y':
            return 'single_with_y'
        elif self.use_outcome == 'y_score':
            return 'single_with_y_score'
        return 'single_without'

# Similarly for MICEImputation, MissForestImputation, MLPImputation, AutoencoderImputation, GAINImputation.
# For methods with multiple imputations (e.g., MICE), set n_imputations=2 in class or pass as param.
# Use rng = default_rng(seed) in each impute method.
# For example, MissForestImputation:
class MissForestImputation(ImputationMethod):
    def __init__(self, use_outcome=None, n_imputations=2):
        self.use_outcome = use_outcome
        self.n_imputations = n_imputations
    
    def impute(self, data, original_data, outcome=None, col_miss=['X1', 'X2'], n_imputations=None, seed=123):
        if n_imputations is None:
            n_imputations = self.n_imputations
        rng = default_rng(seed)
        dat_imputed_list = []
        predictors = [col for col in data.columns if col not in ['y', 'y_score']]
        if self.use_outcome:
            predictors.append(self.use_outcome)
        
        for i in tqdm(range(n_imputations), desc="MissForest Imputations", leave=False):
            # ... (full code from missforest_imputation, using rng.integers for random_state)
            # Return dat_imputed_list
        return dat_imputed_list
    
    @property
    def name(self):
        if self.use_outcome == 'y':
            return 'missforest_with_y'
        elif self.use_outcome == 'y_score':
            return 'missforest_with_y_score'
        return 'missforest_without'

# Implement the rest similarly, with variants for _with_y, _with_y_score, _without by passing use_outcome to __init__.