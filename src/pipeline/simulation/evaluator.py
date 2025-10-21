# evaluator.py
"""Evaluation of imputation quality."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, log_loss
import logging

logger = logging.getLogger(__name__)

def evaluate_imputation(true_data, imputed_list, y='y'):
    """
    Evaluate the quality of imputed data compared to true data for a specific outcome.
    
    Args:
        true_data (pd.DataFrame): Original complete data.
        imputed_list (list): List of imputed DataFrames.
        y (str): Column name for outcome variable to evaluate (default 'y').
    
    Returns:
        dict: Dictionary of evaluation metrics (e.g., MSE, R², log loss) averaged across imputations.
    """
    metrics = {}
    n_imputations = len(imputed_list)
    
    if n_imputations == 0:
        logger.warning("No imputations provided, returning empty metrics.")
        return metrics
    
    # Initialize lists to store metrics across imputations
    mse_values = []
    r2_values = []
    log_loss_values = []
    
    for imputed in imputed_list:
        if y not in true_data.columns or y not in imputed.columns:
            logger.warning(f"Outcome {y} not found in true_data or imputed data, skipping.")
            continue
        
        # Use only complete cases for comparison
        mask = true_data[y].notna() & imputed[y].notna()
        if mask.sum() == 0:
            logger.warning(f"No complete cases for {y}, skipping imputation evaluation.")
            continue
            
        y_true = true_data.loc[mask, y]
        y_imputed = imputed.loc[mask, y]
        
        # Calculate MSE
        mse = mean_squared_error(y_true, y_imputed)
        mse_values.append(mse)
        
        # Fit a model to estimate R² (using true data as baseline)
        model = LinearRegression()
        model.fit(np.arange(len(y_true)).reshape(-1, 1), y_true)
        y_pred_baseline = model.predict(np.arange(len(y_true)).reshape(-1, 1))
        r2 = 1 - (mean_squared_error(y_true, y_imputed) / mean_squared_error(y_true, y_pred_baseline))
        r2_values.append(r2)
        
        # Calculate log loss if y is binary (assuming 0/1 for simplicity)
        if y_true.nunique() == 2 and set(y_true).issubset({0, 1}):
            # Fit logistic regression to get probabilities
            log_reg = LogisticRegression()
            log_reg.fit(np.arange(len(y_true)).reshape(-1, 1), y_true)
            y_pred_prob = log_reg.predict_proba(np.arange(len(y_true)).reshape(-1, 1))[:, 1]
            log_loss_val = log_loss(y_true, y_pred_prob)
            log_loss_values.append(log_loss_val)
    
    # Average metrics across imputations
    if mse_values:
        metrics['mse_mean'] = np.mean(mse_values)
        metrics['mse_std'] = np.std(mse_values) if n_imputations > 1 else 0
    if r2_values:
        metrics['r2_mean'] = np.mean(r2_values)
        metrics['r2_std'] = np.std(r2_values) if n_imputations > 1 else 0
    if log_loss_values:
        metrics['log_loss_mean'] = np.mean(log_loss_values)
        metrics['log_loss_std'] = np.std(log_loss_values) if n_imputations > 1 else 0
    
    return metrics

def evaluate_all_imputations(true_data, imputed_datasets, output_dir):
    results_all = []
    for dataset_name, methods in imputed_datasets.items():
        logger.info(f"Evaluating dataset: {dataset_name}")
        for method_name, imputed_list in methods.items():
            for y in ['y', 'y_score']:
                logger.info(f"Evaluating {dataset_name} - {method_name} - {y}")
                metrics = evaluate_imputation(true_data, imputed_list, y=y)
                result = {
                    'missingness': dataset_name,
                    'method': method_name,
                    'y': y,
                    **metrics
                }
                results_all.append(result)
    
    results_all = pd.DataFrame(results_all)
    return {'results_all': results_all}