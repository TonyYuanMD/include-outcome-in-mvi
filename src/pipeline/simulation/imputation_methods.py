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

# Import GAN models (assuming artifacts/models/gan_models.py exists)
from artifacts.models.gan_models import Generator, Discriminator

logger = logging.getLogger(__name__)

def set_torch_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Set PyTorch determinism for reproducible GPU runs (can slow things down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ImputationMethod(ABC):
    """Abstract base class for imputation methods.
    
    All imputation methods must implement:
    - impute(data, original_data, col_miss=['X1', 'X2'], rng=None): Return list of imputed DataFrames
    - name: Property for descriptive name
    """
    
    @abstractmethod
    def impute(self, data, original_data, col_miss=['X1', 'X2'], rng=None): # Removed outcome and n_imputations
        pass
    
    @property
    @abstractmethod
    def name(self):
        pass

class CompleteData(ImputationMethod):
    def impute(self, data, original_data, col_miss=['X1', 'X2'], n_imputations=1, rng=None):
        dat_complete = data.dropna(subset=col_miss)
        if len(dat_complete) == 0:
            logger.warning(f"No complete cases remain after dropping rows with missing values in {col_miss}. Returning original data unchanged.")
            return [original_data.copy()]  # Return original data without imputation
        return [dat_complete]
    
    @property
    def name(self):
        return 'complete_data'

class MeanImputation(ImputationMethod):
    def impute(self, data, original_data, col_miss=['X1', 'X2'], n_imputations=1, rng=None):
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

class SingleImputation(ImputationMethod):
    def __init__(self, use_outcome=None):
        self.use_outcome = use_outcome  # 'y', 'y_score', or None
    
    def impute(self, data, original_data, col_miss=['X1', 'X2'], rng=None):
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

class MICEImputation(ImputationMethod):
    def __init__(self, use_outcome=None, n_imputations=5):
        self.use_outcome = use_outcome
        self.n_imputations = n_imputations
    
    def impute(self, data, original_data, col_miss=['X1', 'X2'], rng=None):
        if rng is None:
            rng = default_rng(123)
        dat_imputed_list = []
        predictors = [col for col in data.columns if col not in col_miss + ['y', 'y_score']]  # Exclude col_miss to avoid duplicates
        if self.use_outcome:
            predictors.append(self.use_outcome)
        imputation_rngs = rng.spawn(self.n_imputations)
        
        for i in tqdm(range(self.n_imputations), desc="MICE Imputations", leave=False):
            imputation_rng = imputation_rngs[i]
            imp = IterativeImputer(max_iter=10, random_state=imputation_rng.integers(0, 2**32), sample_posterior=True)
            dat_imputed = data.copy()
            X = dat_imputed[predictors + col_miss]
            X_imputed = pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index)
            dat_imputed[col_miss] = X_imputed[col_miss]
            for out in ['y', 'y_score']:
                if out in original_data.columns and out not in dat_imputed.columns:
                    dat_imputed[out] = original_data[out]
            dat_imputed_list.append(dat_imputed)
        return dat_imputed_list
    
    @property
    def name(self):
        if self.use_outcome == 'y':
            return 'mice_with_y'
        elif self.use_outcome == 'y_score':
            return 'mice_with_y_score'
        return 'mice_without'

class MissForestImputation(ImputationMethod):
    def __init__(self, use_outcome=None, n_imputations=5):
        self.use_outcome = use_outcome
        self.n_imputations = n_imputations
    
    def impute(self, data, original_data, col_miss=['X1', 'X2'], rng=None):
        if rng is None:
            rng = default_rng(123)
        dat_imputed_list = []
        predictors = [col for col in data.columns if col not in col_miss + ['y', 'y_score']]  # Exclude col_miss to avoid duplicates
        if self.use_outcome:
            predictors.append(self.use_outcome)
        imputation_rngs = rng.spawn(self.n_imputations)
        for i in tqdm(range(self.n_imputations), desc="MissForest Imputations", leave=False):
            imputation_rng = imputation_rngs[i]
            dat_imputed = data.copy()
            for col in col_miss:
                mask = ~data[col].isna()
                X_train = dat_imputed.loc[mask, predictors]
                y_train = data.loc[mask, col]
                if X_train.isna().any().any():
                    X_train = X_train.fillna(X_train.mean())
                model = RandomForestRegressor(n_estimators=100, random_state=imputation_rng.integers(0, 10000))
                model.fit(X_train, y_train)
                mask_missing = data[col].isna()
                X_missing = dat_imputed.loc[mask_missing, predictors]
                if len(X_missing) > 0:
                    if X_missing.isna().any().any():
                        X_missing = X_missing.fillna(X_missing.mean())
                    dat_imputed.loc[mask_missing, col] = model.predict(X_missing)
            for out in ['y', 'y_score']:
                if out in original_data.columns and out not in dat_imputed.columns:
                    dat_imputed[out] = original_data[out]
            dat_imputed_list.append(dat_imputed)
        return dat_imputed_list
    
    @property
    def name(self):
        if self.use_outcome == 'y':
            return 'missforest_with_y'
        elif self.use_outcome == 'y_score':
            return 'missforest_with_y_score'
        return 'missforest_without'

class MLPImputation(ImputationMethod):
    def __init__(self, use_outcome=None, n_imputations=5):
        self.use_outcome = use_outcome
        self.n_imputations = n_imputations
    
    def impute(self, data, original_data, col_miss=['X1', 'X2'], rng=None):
        if rng is None:
            rng = default_rng(123)
        dat_imputed_list = []
        predictors = [col for col in data.columns if col not in col_miss + ['y', 'y_score']]  # Exclude col_miss to avoid duplicates
        if self.use_outcome:
            predictors.append(self.use_outcome)
        scaler = StandardScaler()
        imputation_rngs = rng.spawn(self.n_imputations)
        for i in tqdm(range(self.n_imputations), desc="MLP Imputations", leave=False):
            imputation_rng = imputation_rngs[i]
            dat_imputed = data.copy()
            for col in col_miss:
                mask = ~data[col].isna()
                X_train = dat_imputed.loc[mask, predictors]
                y_train = data.loc[mask, col]
                if X_train.isna().any().any():
                    X_train = X_train.fillna(X_train.mean())
                X_train_scaled = scaler.fit_transform(X_train)
                model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=imputation_rng.integers(0, 10000))
                model.fit(X_train_scaled, y_train)
                mask_missing = data[col].isna()
                X_missing = dat_imputed.loc[mask_missing, predictors]
                if len(X_missing) > 0:
                    if X_missing.isna().any().any():
                        X_missing = X_missing.fillna(X_missing.mean())
                    X_missing_scaled = scaler.transform(X_missing)
                    dat_imputed.loc[mask_missing, col] = model.predict(X_missing_scaled)
            for out in ['y', 'y_score']:
                if out in original_data.columns and out not in dat_imputed.columns:
                    dat_imputed[out] = original_data[out]
            dat_imputed_list.append(dat_imputed)
        return dat_imputed_list
    
    @property
    def name(self):
        if self.use_outcome == 'y':
            return 'mlp_with_y'
        elif self.use_outcome == 'y_score':
            return 'mlp_with_y_score'
        return 'mlp_without'

class AutoencoderImputation(ImputationMethod):
    def __init__(self, use_outcome=None, n_imputations=5, hidden_dims=(64, 32, 64)):
        self.use_outcome = use_outcome
        self.n_imputations = n_imputations
        self.hidden_dims = hidden_dims
    
    def impute(self, data, original_data, col_miss=['X1', 'X2'], rng=None):
        if rng is None:
            rng = default_rng(123)
        dat_imputed_list = []
        predictors = [col for col in data.columns if col not in col_miss + ['y', 'y_score']]
        if self.use_outcome:
            predictors.append(self.use_outcome)
        n_features = len(predictors) + len(col_miss)
        class Autoencoder(nn.Module):
            def __init__(self, input_dim, hidden_dims):
                super(Autoencoder, self).__init__()
                layers = []
                dims = [input_dim] + list(hidden_dims)
                for i in range(len(dims) - 1):
                    layers.append(nn.Linear(dims[i], dims[i + 1]))
                    layers.append(nn.ReLU())
                layers.append(nn.Linear(dims[-1], input_dim))
                self.encoder = nn.Sequential(*layers[:len(layers)//2])
                self.decoder = nn.Sequential(*layers[len(layers)//2:])
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Autoencoder(n_features, self.hidden_dims).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scaler = StandardScaler()
        imputation_rngs = rng.spawn(self.n_imputations)
        for i in tqdm(range(self.n_imputations), desc="Autoencoder Imputations", leave=False):
            imputation_rng = imputation_rngs[i]
            dat_imputed = data.copy()
            X = dat_imputed[predictors + col_miss].fillna(0)  # Initial imputation
            set_torch_seed(imputation_rng.integers(0, 2**32))
            X_scaled = scaler.fit_transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            for epoch in range(100):
                model.train()
                optimizer.zero_grad()
                output = model(X_tensor)
                loss = criterion(output, X_tensor)
                loss.backward()
                optimizer.step()
            mask_missing = data[col_miss].isna().any(axis=1)
            X_missing = dat_imputed.loc[mask_missing, predictors + col_miss].fillna(0)
            X_missing_scaled = scaler.transform(X_missing)
            X_missing_tensor = torch.FloatTensor(X_missing_scaled).to(device)
            with torch.no_grad():
                X_missing_imputed_scaled = model(X_missing_tensor)
            X_missing_imputed = scaler.inverse_transform(X_missing_imputed_scaled.cpu().numpy())
            dat_imputed.loc[mask_missing, col_miss] = X_missing_imputed[:, len(predictors):]
            for out in ['y', 'y_score']:
                if out in original_data.columns and out not in dat_imputed.columns:
                    dat_imputed[out] = original_data[out]
            dat_imputed_list.append(dat_imputed)
        return dat_imputed_list
    
    @property
    def name(self):
        if self.use_outcome == 'y':
            return 'autoencoder_with_y'
        elif self.use_outcome == 'y_score':
            return 'autoencoder_with_y_score'
        return 'autoencoder_without'

class GAINImputation(ImputationMethod):
    def __init__(self, use_outcome=None, n_imputations=5, alpha=100, hidden_dim=128):
        self.use_outcome = use_outcome
        self.n_imputations = n_imputations
        self.alpha = alpha
        self.hidden_dim = hidden_dim
    
    # In imputation_methods.py, within GAINImputation class
    def impute(self, data, original_data, col_miss=['X1', 'X2'], rng=None):
        if rng is None:
            rng = default_rng(123)
        dat_imputed_list = []
        predictors = [col for col in data.columns if col not in col_miss + ['y', 'y_score']]
        if self.use_outcome:
            if self.use_outcome in original_data.columns:
                predictors.append(self.use_outcome)
            else:
                logger.warning(f"Outcome {self.use_outcome} not found in original_data, ignoring.")
        n_features = len(predictors) + len(col_miss)  # Full features
        class GAIN(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super(GAIN, self).__init__()
                self.generator = Generator(input_dim, hidden_dim)
                self.discriminator = Discriminator(input_dim, hidden_dim)
            
            def forward(self, x, m):
                g_out = self.generator(x, m)
                d_out = self.discriminator(x, m, g_out)
                return g_out, d_out
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GAIN(n_features, self.hidden_dim).to(device)
        criterion = nn.BCELoss()  # Define but use manually for GAIN-specific loss
        optimizer_g = optim.Adam(model.generator.parameters(), lr=0.001)
        optimizer_d = optim.Adam(model.discriminator.parameters(), lr=0.001)
        scaler = StandardScaler()
        imputation_rngs = rng.spawn(self.n_imputations)
        for i in tqdm(range(self.n_imputations), desc="GAIN Imputations", leave=False):
            imputation_rng = imputation_rngs[i]
            dat_imputed = data.copy()
            X = dat_imputed[predictors + col_miss].fillna(0)
            set_torch_seed(imputation_rng.integers(0, 2**32))
            X_scaled = scaler.fit_transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            M = torch.FloatTensor(1 - data[col_miss].isna().astype(int).values).to(device)  # Shape (50, 2)
            # Create full mask: 1 for predictors (observed), M for col_miss
            M_full = torch.ones_like(X_tensor)  # Shape (50, n_features)
            M_full[:, len(predictors):] = M  # Set last len(col_miss) columns to the actual mask
            for epoch in range(1000):
                model.train()
                # Generator step
                optimizer_g.zero_grad()
                g_out, d_out = model(X_tensor, M_full)
                g_loss = -torch.mean(torch.log(d_out + 1e-8)) + self.alpha * torch.mean(torch.abs(M_full[:, len(predictors):] * (X_tensor[:, len(predictors):] - g_out[:, len(predictors):])))
                g_loss.backward(retain_graph=True)  # Retain graph for discriminator
                optimizer_g.step()
                # Discriminator step
                optimizer_d.zero_grad()
                d_out = model.discriminator(X_tensor, M_full, g_out.detach())  # Detach g_out to avoid double gradients
                d_loss = -torch.mean(torch.log(d_out + 1e-8) + torch.log(1 - d_out + 1e-8))
                d_loss.backward()
                optimizer_d.step()
            mask_missing = data[col_miss].isna().any(axis=1)
            X_missing = dat_imputed.loc[mask_missing, predictors + col_miss].fillna(0)
            X_missing_scaled = scaler.transform(X_missing)
            X_missing_tensor = torch.FloatTensor(X_missing_scaled).to(device)
            M_missing = torch.FloatTensor(1 - X_missing[col_miss].isna().astype(int).values).to(device)
            M_missing_full = torch.ones_like(X_missing_tensor)
            M_missing_full[:, len(predictors):] = M_missing
            with torch.no_grad():
                g_out_missing, _ = model(X_missing_tensor, M_missing_full)
            X_missing_imputed = scaler.inverse_transform(g_out_missing.cpu().numpy())
            dat_imputed.loc[mask_missing, col_miss] = X_missing_imputed[:, len(predictors):]  # Slice the imputed part
            for out in ['y', 'y_score']:
                if out in original_data.columns and out not in dat_imputed.columns:
                    dat_imputed[out] = original_data[out]
            dat_imputed_list.append(dat_imputed)
        return dat_imputed_list
        
    @property
    def name(self):
        if self.use_outcome == 'y':
            return 'gain_with_y'
        elif self.use_outcome == 'y_score':
            return 'gain_with_y_score'
        return 'gain_without'