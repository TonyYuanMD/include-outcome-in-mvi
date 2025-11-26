"""Evaluation of imputation quality.

This module provides functions to evaluate the utility of imputed datasets
by training downstream prediction models on imputed training data and
evaluating them on complete test data.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, log_loss
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# NUMERICAL STABILITY FUNCTIONS
# ============================================================================

def stable_log_loss(y_true, y_pred_proba, eps=1e-15):
    """
    Compute log loss with numerical stability.
    
    Uses clipping to prevent log(0) and log(1) issues.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred_proba : array-like
        Predicted probabilities
    eps : float
        Small value for clipping probabilities
        
    Returns:
    --------
    float : Log loss value
    """
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))

def stable_variance(values, ddof=0):
    """
    Compute variance with numerical stability using two-pass algorithm.
    
    More stable than np.var for large datasets or when values have large range.
    
    Parameters:
    -----------
    values : array-like
        Array of values
    ddof : int
        Delta degrees of freedom (0 for population variance, 1 for sample)
        
    Returns:
    --------
    float : Variance value
    """
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return 0.0
    
    values = np.asarray(values, dtype=np.float64)
    mean_val = np.mean(values)
    # Two-pass algorithm for numerical stability
    variance = np.mean((values - mean_val) ** 2)
    
    # Adjust for degrees of freedom
    if ddof > 0 and len(values) > ddof:
        variance = variance * len(values) / (len(values) - ddof)
    
    return float(variance)

def stable_std(values, ddof=0):
    """
    Compute standard deviation with numerical stability.
    
    Parameters:
    -----------
    values : array-like
        Array of values
    ddof : int
        Delta degrees of freedom
        
    Returns:
    --------
    float : Standard deviation value
    """
    variance = stable_variance(values, ddof=ddof)
    return np.sqrt(max(0.0, variance))  # Ensure non-negative

def evaluate_imputation(imputed_list, test_data, y='y'):
    """
    Evaluate the utility of imputed data using a downstream prediction model
    (LinearRegression for y_score, LogisticRegression for y) evaluated on complete test data.
    
    Args:
        imputed_list (list): List of imputed TRAINING DataFrames.
        test_data (pd.DataFrame): Complete TEST Data (Ground Truth).
        y (str): Column name for outcome variable ('y' or 'y_score').
    
    Returns:
        dict: Dictionary of evaluation metrics averaged across imputations.
    """
    metrics = {}
    n_imputations = len(imputed_list)
    
    if n_imputations == 0:
        logger.warning("No imputations provided, returning empty metrics.")
        return metrics

    # Predictor columns: all columns present in data except both outcomes
    outcome_cols = ['y', 'y_score']
    predictors = [col for col in test_data.columns if col not in outcome_cols]
    
    if not all(col in test_data.columns for col in predictors + [y]):
        logger.error(f"Test data missing required columns for {y} prediction.")
        return metrics

    # Prepare Test Data (vectorized - compute once)
    X_test = test_data[predictors].values  # Convert to numpy array for faster operations
    y_test = test_data[y].values
    
    # Pre-check for NaNs in test data (once, not in loop)
    if np.isnan(X_test).any():
        logger.warning("NaNs detected in X_test. Skipping evaluation.")
        return metrics
    
    # Initialize arrays to store metrics across imputations (vectorized storage)
    mse_values = []
    log_loss_values = []
    r2_values = []

    # OPTIMIZATION: Batch process imputations where possible
    for imputed_train in imputed_list:
        X_train = imputed_train[predictors].values  # Convert to numpy array
        y_train = imputed_train[y].values
        
        # --- Safeguard for stability: Imputed data should ideally be clean ---
        if np.isnan(X_train).any():
             logger.warning("NaNs detected in imputed covariates X_train. Skipping this imputation run.")
             continue
        # --------------------------------------------------------------------
        
        if y == 'y_score': # Continuous outcome -> Linear Regression
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics for continuous outcome (vectorized)
            mse = np.mean((y_test - y_pred) ** 2)  # Direct calculation, faster than sklearn
            mse_values.append(mse)
            
            # R² calculation (vectorized)
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-10))  # Add small epsilon to prevent division by zero
            r2_values.append(r2)
            
        elif y == 'y': # Binary outcome -> Logistic Regression
            # Use liblinear for convergence and simpler models, set max_iter for stability
            model = LogisticRegression(solver='liblinear', random_state=123, max_iter=100)
            try:
                model.fit(X_train, y_train)
            except Exception as e:
                logger.error(f"LogisticRegression fit failed: {e}")
                continue # Skip if model fails to fit
            
            # Use predict_proba for Log Loss
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate Log Loss with numerical stability
            log_loss_val = stable_log_loss(y_test, y_pred_proba)
            log_loss_values.append(log_loss_val)
    
    # Average metrics across imputations using stable calculations
    if mse_values:
        metrics['mse_mean'] = np.mean(mse_values)
        metrics['mse_std'] = stable_std(mse_values, ddof=0) if n_imputations > 1 else 0.0
    if r2_values:
        metrics['r2_mean'] = np.mean(r2_values)
        metrics['r2_std'] = stable_std(r2_values, ddof=0) if n_imputations > 1 else 0.0
    if log_loss_values:
        metrics['log_loss_mean'] = np.mean(log_loss_values)
        metrics['log_loss_std'] = stable_std(log_loss_values, ddof=0) if n_imputations > 1 else 0.0
    
    return metrics

def evaluate_all_imputations(true_data, imputed_datasets, output_dir=None):
    """
    Evaluate multiple imputation methods across multiple missingness patterns.
    
    This is a legacy function for batch evaluation. The recommended approach
    is to use `evaluate_imputation` directly within the simulation framework.
    
    Parameters:
    -----------
    true_data : pd.DataFrame
        Complete ground truth data
    imputed_datasets : dict
        Nested dictionary: {missingness_pattern: {method_name: [imputed_dfs]}}
    output_dir : str, optional
        Directory to save results (currently not used)
    
    Returns:
    --------
    dict : Dictionary with 'results_all' key containing a DataFrame of results
    """
    results_all = []
    for dataset_name, methods in imputed_datasets.items():
        logger.info(f"Evaluating dataset: {dataset_name}")
        for method_name, imputed_list in methods.items():
            for y in ['y', 'y_score']:
                logger.info(f"Evaluating {dataset_name} - {method_name} - {y}")
                metrics = evaluate_imputation(imputed_list, true_data, y=y)
                result = {
                    'missingness': dataset_name,
                    'method': method_name,
                    'y': y,
                    **metrics
                }
                results_all.append(result)
    
    results_all = pd.DataFrame(results_all)
    return {'results_all': results_all}