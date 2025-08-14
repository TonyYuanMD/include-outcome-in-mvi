import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import logging
from tqdm import tqdm

# Import GAN models
from gan_models import Generator, Discriminator

# Configure logging
logger = logging.getLogger(__name__)

def missforest_imputation(data, original_data, outcome=None, col_miss=['X1', 'X2'], n_imputations=5, seed=123):
    """
    Impute missing values using Random Forest (MissForest-like) with iterative mean initialization.
    """
    np.random.seed(seed)
    dat_imputed_list = []
    predictors = [col for col in data.columns if col not in ['y', 'y_score']]
    if outcome:
        predictors.append(outcome)
    
    for i in tqdm(range(n_imputations), desc="MissForest Imputations", leave=False):
        # Initialize with mean imputation
        dat_imputed = data.copy()
        for col in col_miss:
            dat_imputed[col] = dat_imputed[col].fillna(dat_imputed[col].mean())
        
        # Iterate until convergence or max iterations
        max_iter = 10
        tol = 0.001
        logger.info(f"MissForest imputation {i+1}/{n_imputations}: Starting iterations")
        for iteration in range(max_iter):
            old_imputed = dat_imputed[col_miss].copy()
            for col in col_miss:
                mask = ~data[col].isna()
                if mask.sum() < 50:
                    warnings.warn(f"Too few complete cases ({mask.sum()}) for {col} in MissForest, iteration {iteration}, keeping mean imputation")
                    logger.warning(f"MissForest: Too few complete cases for {col}, iteration {iteration}")
                    continue
                X_train = dat_imputed.loc[mask, [p for p in predictors if p != col]]
                y_train = data.loc[mask, col]
                model = RandomForestRegressor(n_estimators=100, random_state=seed + i + iteration)
                model.fit(X_train, y_train)
                mask_missing = data[col].isna()
                X_missing = dat_imputed.loc[mask_missing, [p for p in predictors if p != col]]
                if len(X_missing) > 0:
                    dat_imputed.loc[mask_missing, col] = model.predict(X_missing)
            # Check convergence
            diff = ((dat_imputed[col_miss] - old_imputed) ** 2).mean().mean()
            logger.info(f"MissForest imputation {i+1}, iteration {iteration+1}: Convergence diff = {diff:.6f}")
            if diff < tol:
                logger.info(f"MissForest imputation {i+1}: Converged at iteration {iteration+1}")
                break
        # Restore outcomes
        for out in ['y', 'y_score']:
            if out in original_data.columns and out not in dat_imputed.columns:
                dat_imputed[out] = original_data[out]
        dat_imputed_list.append(dat_imputed)
    return dat_imputed_list

def mlp_imputation(data, original_data, outcome=None, col_miss=['X1', 'X2'], n_imputations=5, seed=123):
    """
    Impute missing values using MLP Regressor with iterative mean initialization and scaling.
    """
    np.random.seed(seed)
    dat_imputed_list = []
    predictors = [col for col in data.columns if col not in ['y', 'y_score']]
    if outcome:
        predictors.append(outcome)
    
    for i in tqdm(range(n_imputations), desc="MLP Imputations", leave=False):
        # Initialize with mean imputation
        dat_imputed = data.copy()
        for col in col_miss:
            dat_imputed[col] = dat_imputed[col].fillna(dat_imputed[col].mean())
        
        # Iterate until convergence or max iterations
        max_iter = 10
        tol = 0.001
        logger.info(f"MLP imputation {i+1}/{n_imputations}: Starting iterations")
        for iteration in range(max_iter):
            old_imputed = dat_imputed[col_miss].copy()
            for col in col_miss:
                mask = ~data[col].isna()
                if mask.sum() < 50:
                    warnings.warn(f"Too few complete cases ({mask.sum()}) for {col} in MLP, iteration {iteration}, keeping mean imputation")
                    logger.warning(f"MLP: Too few complete cases for {col}, iteration {iteration}")
                    continue
                X_train = dat_imputed.loc[mask, [p for p in predictors if p != col]]
                y_train = data.loc[mask, col]
                # Scale predictors
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                model = MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    max_iter=1000,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=10,
                    random_state=seed + i + iteration
                )
                model.fit(X_train_scaled, y_train)
                mask_missing = data[col].isna()
                X_missing = dat_imputed.loc[mask_missing, [p for p in predictors if p != col]]
                if len(X_missing) > 0:
                    X_missing_scaled = scaler.transform(X_missing)
                    dat_imputed.loc[mask_missing, col] = model.predict(X_missing_scaled)
            # Check convergence
            diff = ((dat_imputed[col_miss] - old_imputed) ** 2).mean().mean()
            logger.info(f"MLP imputation {i+1}, iteration {iteration+1}: Convergence diff = {diff:.6f}")
            if diff < tol:
                logger.info(f"MLP imputation {i+1}: Converged at iteration {iteration+1}")
                break
        # Restore outcomes
        for out in ['y', 'y_score']:
            if out in original_data.columns and out not in dat_imputed.columns:
                dat_imputed[out] = original_data[out]
        dat_imputed_list.append(dat_imputed)
    return dat_imputed_list

def autoencoder_imputation(data, original_data, outcome=None, col_miss=['X1', 'X2'], n_imputations=5, seed=123):
    """
    Impute missing values using an Autoencoder (PyTorch) with GPU support if available.
    """
    torch.manual_seed(seed)
    dat_imputed_list = []
    predictors = [col for col in data.columns if col not in ['y', 'y_score']]
    if outcome:
        predictors.append(outcome)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Autoencoder using device: {device}")
    
    # Normalize data
    data_norm = data[predictors].copy()
    means = data_norm.mean()
    stds = data_norm.std().replace(0, 1)
    data_norm = (data_norm - means) / stds
    data_norm = data_norm.fillna(0)
    
    class Autoencoder(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, input_dim)
            )
        
        def forward(self, x):
            return self.decoder(self.encoder(x))
    
    input_dim = len(predictors)
    for i in tqdm(range(n_imputations), desc="AE Imputations", leave=False):
        model = Autoencoder(input_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        data_tensor = torch.tensor(data_norm.values, dtype=torch.float32).to(device)
        
        # Train with early stopping
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        logger.info(f"AE imputation {i+1}/{n_imputations}: Starting training")
        for epoch in range(50):
            optimizer.zero_grad()
            output = model(data_tensor)
            loss = criterion(output, data_tensor)
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            if loss_value < best_loss:
                best_loss = loss_value
                patience_counter = 0
            else:
                patience_counter += 1
            logger.info(f"AE imputation {i+1}, epoch {epoch+1}: Loss = {loss_value:.6f}")
            if patience_counter >= patience:
                logger.info(f"AE imputation {i+1}: Early stopping at epoch {epoch+1}")
                break
        
        # Impute
        with torch.no_grad():
            imputed_norm = model(data_tensor).cpu().numpy()
        imputed_data = pd.DataFrame(imputed_norm, columns=predictors, index=data.index)
        imputed_data = imputed_data * stds + means
        dat_imputed = data.copy()
        for col in col_miss:
            mask_missing = data[col].isna()
            dat_imputed.loc[mask_missing, col] = imputed_data.loc[mask_missing, col]
        for out in ['y', 'y_score']:
            if out in original_data.columns and out not in dat_imputed.columns:
                dat_imputed[out] = original_data[out]
        dat_imputed_list.append(dat_imputed)
    return dat_imputed_list

def gain_imputation(data, original_data, outcome=None, col_miss=['X1', 'X2'], n_imputations=5, seed=123):
    """
    Impute missing values using a GAIN-like GAN in PyTorch with GPU support if available.
    """
    torch.manual_seed(seed)
    dat_imputed_list = []
    predictors = [col for col in data.columns if col not in ['y', 'y_score']]
    if outcome:
        predictors.append(outcome)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"GAIN using device: {device}")
    
    # Normalize data
    data_norm = data[predictors].copy()
    means = data_norm.mean()
    stds = data_norm.std().replace(0, 1)
    data_norm = (data_norm - means) / stds
    mask = data_norm.isna().astype(float)
    data_norm = data_norm.fillna(0)
    n_samples, input_dim = data_norm.shape
    
    for i in tqdm(range(n_imputations), desc="GAN Imputations", leave=False):
        generator = Generator(input_dim).to(device)
        discriminator = Discriminator(input_dim).to(device)
        g_optimizer = optim.Adam(generator.parameters(), lr=0.001)
        d_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCELoss()
        
        data_tensor = torch.tensor(data_norm.values, dtype=torch.float32).to(device)
        mask_tensor = torch.tensor(mask.values, dtype=torch.float32).to(device)
        
        # Training loop with early stopping
        batch_size = 32
        steps = max(1, n_samples // batch_size)
        best_g_loss = float('inf')
        patience = 10
        patience_counter = 0
        logger.info(f"GAN imputation {i+1}/{n_imputations}: Starting training")
        for epoch in range(50):
            epoch_g_loss = 0
            for _ in range(steps):
                idx = torch.randint(0, n_samples, (batch_size,))
                real_data = data_tensor[idx]
                real_mask = mask_tensor[idx]
                
                # Train Discriminator
                gen_input = torch.cat([real_data, real_mask], dim=1)
                gen_data = generator(gen_input)
                imputed = real_data * (1 - real_mask) + gen_data * real_mask
                
                d_real_input = torch.cat([real_data, real_mask], dim=1)
                d_fake_input = torch.cat([imputed.detach(), real_mask], dim=1)
                
                d_real = discriminator(d_real_input)
                d_fake = discriminator(d_fake_input)
                
                d_loss_real = bce_loss(d_real, torch.ones_like(d_real) * 0.9)
                d_loss_fake = bce_loss(d_fake, torch.zeros_like(d_fake))
                d_loss = d_loss_real + d_loss_fake
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()
                
                # Train Generator
                gen_input = torch.cat([real_data, real_mask], dim=1)
                gen_data = generator(gen_input)
                imputed = real_data * (1 - real_mask) + gen_data * real_mask
                d_fake_input = torch.cat([imputed, real_mask], dim=1)
                d_fake = discriminator(d_fake_input)
                
                g_adv_loss = bce_loss(d_fake, torch.ones_like(d_fake))
                g_mse_loss = mse_loss(gen_data * (1 - real_mask), real_data * (1 - real_mask))
                g_loss = 10 * g_mse_loss + g_adv_loss
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()
                
                epoch_g_loss += g_loss.item()
            
            # Early stopping
            epoch_g_loss /= steps
            logger.info(f"GAN imputation {i+1}, epoch {epoch+1}: Generator loss = {epoch_g_loss:.6f}")
            if epoch_g_loss < best_g_loss:
                best_g_loss = epoch_g_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"GAN imputation {i+1}: Early stopping at epoch {epoch+1}")
                break
        
        # Impute
        gen_input = torch.cat([data_tensor, mask_tensor], dim=1)
        with torch.no_grad():
            imputed_norm = generator(gen_input).cpu().numpy()
        imputed_data = pd.DataFrame(imputed_norm, columns=predictors, index=data.index)
        imputed_data = imputed_data * stds + means
        dat_imputed = data.copy()
        for col in col_miss:
            mask_missing = data[col].isna()
            dat_imputed.loc[mask_missing, col] = imputed_data.loc[mask_missing, col]
        for out in ['y', 'y_score']:
            if out in original_data.columns and out not in dat_imputed.columns:
                dat_imputed[out] = original_data[out]
        dat_imputed_list.append(dat_imputed)
    return dat_imputed_list

# Documentation
"""
Functions:
- missforest_imputation: Imputes X1, X2 with Random Forest (5 imputations), using iterative mean initialization
- mlp_imputation: Imputes X1, X2 with MLP Regressor (5 imputations), using iterative mean initialization and scaling
- autoencoder_imputation: Imputes X1, X2 with Autoencoder (5 imputations, PyTorch, GPU support)
- gain_imputation: Imputes X1, X2 with GAIN-like GAN (5 imputations, PyTorch, GPU support, imports models from gan_models.py)
"""