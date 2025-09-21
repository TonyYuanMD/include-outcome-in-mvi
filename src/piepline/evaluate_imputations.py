import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, log_loss
import logging
import os

logger = logging.getLogger(__name__)

def evaluate_imputation(true_data, imputed_data, y='y'):
    """
    Evaluate imputation quality for a single dataset.
    
    Parameters:
    - true_data: Original complete DataFrame
    - imputed_data: List of imputed DataFrames
    - y: Response variable ('y' or 'y_score')
    
    Returns:
    - results: Dictionary of evaluation metrics
    """
    results = {}
    predictors = [col for col in true_data.columns if col not in ['y', 'y_score']]
    
    if y == 'y_score':
        model = LinearRegression()
        metrics = ['rmse', 'bias']
    else:
        model = LogisticRegression(max_iter=1000)
        metrics = ['log_loss', 'bias']
    
    if len(imputed_data) == 1:  # Single imputation (mean, single, complete_data)
        dat = imputed_data[0]
        X_true = true_data[predictors]
        y_true = true_data[y]
        X_imputed = dat[predictors]
        y_imputed = dat[y]
        
        # Check for NaNs in X_imputed
        if X_imputed.isna().any().any():
            nan_cols = X_imputed.columns[X_imputed.isna().any()].tolist()
            logger.error(f"NaN values found in X_imputed for y={y}, columns: {nan_cols}")
            # Fallback: Fill NaNs with mean for evaluation
            X_imputed = X_imputed.fillna(X_imputed.mean())
            logger.warning(f"Filled NaNs with column means for evaluation: {nan_cols}")
        
        model.fit(X_imputed, y_imputed)
        y_pred = model.predict(X_true)
        if y == 'y_score':
            results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            results['bias'] = np.mean(y_pred - y_true)
        else:
            y_pred_proba = model.predict_proba(X_true)[:, 1]
            results['log_loss'] = log_loss(y_true, y_pred_proba)
            results['bias'] = np.mean(y_pred_proba - y_true)
    else:  # Multiple imputations (MICE, MissForest, MLP, AE, GAN)
        estimates = []
        for dat in imputed_data:
            X_imputed = dat[predictors]
            y_imputed = dat[y]
            
            # Check for NaNs in X_imputed
            if X_imputed.isna().any().any():
                nan_cols = X_imputed.columns[X_imputed.isna().any()].tolist()
                logger.error(f"NaN values found in X_imputed for y={y}, columns: {nan_cols}")
                X_imputed = X_imputed.fillna(X_imputed.mean())
                logger.warning(f"Filled NaNs with column means for evaluation: {nan_cols}")
            
            model.fit(X_imputed, y_imputed)
            y_pred = model.predict(true_data[predictors])
            estimates.append(y_pred)
        
        # Pool predictions (mean for point estimates)
        pooled_pred = np.mean(estimates, axis=0)
        if y == 'y_score':
            results['rmse'] = np.sqrt(mean_squared_error(true_data[y], pooled_pred))
            results['bias'] = np.mean(pooled_pred - true_data[y])
        else:
            # Approximate log_loss for pooled binary predictions
            pooled_pred_proba = np.mean([model.predict_proba(true_data[predictors])[:, 1] for model in [LogisticRegression(max_iter=1000).fit(dat[predictors], dat[y]) for dat in imputed_data]], axis=0)
            results['log_loss'] = log_loss(true_data[y], pooled_pred_proba)
            results['bias'] = np.mean(pooled_pred_proba - true_data[y])
    
    return results

def evaluate_all_imputations(true_data, imputed_datasets, output_dir):
    """
    Evaluate all imputed datasets.
    
    Parameters:
    - true_data: Original complete DataFrame
    - imputed_datasets: Dictionary of imputed datasets
    - output_dir: Directory to save results
    
    Returns:
    - results: Dictionary with evaluation results
    """
    results_all = []
    for dataset_name, methods in imputed_datasets.items():
        logger.info(f"Evaluating dataset: {dataset_name}")
        for method, imputed in methods.items():
            for y in ['y', 'y_score']:
                logger.info(f"Evaluating {dataset_name} - {method} - {y}")
                metrics = evaluate_imputation(true_data, imputed['full'], y=y)
                result = {
                    'missingness': dataset_name,
                    'method': method,
                    'y': y,
                    **metrics
                }
                results_all.append(result)
    
    results_all = pd.DataFrame(results_all)
    results_all.to_csv(os.path.join(output_dir, 'evaluation_results.csv'), index=False)
    logger.info(f"Saved evaluation results to {output_dir}")
    return {'results_all': results_all}

# Documentation
"""
Functions:
- evaluate_imputation: Evaluates imputation quality for a single dataset
- evaluate_all_imputations: Evaluates all imputed datasets and saves results
"""