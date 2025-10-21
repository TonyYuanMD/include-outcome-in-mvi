# evaluator.py
"""Evaluation of imputation quality."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, log_loss
import logging

logger = logging.getLogger(__name__)

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

    # Prepare Test Data
    X_test = test_data[predictors]
    y_test = test_data[y]
    
    # Initialize lists to store metrics across imputations
    mse_values = []
    log_loss_values = []
    r2_values = []

    for imputed_train in imputed_list:
        X_train = imputed_train[predictors]
        y_train = imputed_train[y]
        
        # --- Safeguard for stability: Imputed data should ideally be clean ---
        if X_train.isna().any().any():
             logger.warning("NaNs detected in imputed covariates X_train. Skipping this imputation run.")
             continue
        if X_test.isna().any().any():
            logger.warning("NaNs detected in X_test. Skipping this imputation run.")
            continue
        # --------------------------------------------------------------------
        
        if y == 'y_score': # Continuous outcome -> Linear Regression
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics for continuous outcome
            mse_values.append(mean_squared_error(y_test, y_pred))
            r2_values.append(model.score(X_test, y_test))
            
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
            
            # Calculate Log Loss
            log_loss_values.append(log_loss(y_test, y_pred_proba))
            
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