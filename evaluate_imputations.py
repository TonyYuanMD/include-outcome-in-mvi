import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error

def pool_rubin(coefficients, variances, m):
    """
    Pool coefficients using Rubin's rules.
    
    Parameters:
    - coefficients: List of coefficient arrays
    - variances: List of variance arrays
    - m: Number of imputations
    
    Returns:
    - pooled_beta: Pooled coefficients
    - pooled_se: Pooled standard errors
    """
    coefficients = np.array(coefficients)
    variances = np.array(variances)
    pooled_beta = np.mean(coefficients, axis=0)
    within_var = np.mean(variances, axis=0)
    between_var = np.var(coefficients, axis=0, ddof=1)
    total_var = within_var + between_var * (1 + 1/m)
    pooled_se = np.sqrt(total_var)
    return pooled_beta, pooled_se

def evaluate_imputation(full_list, original_data, pre_imputation_data, eval_data, true_beta, predictor_names, outcome='y', col_miss=['X1', 'X2']):
    """
    Evaluate imputation methods.
    
    Parameters:
    - full_list: List of imputed DataFrames
    - original_data: Original complete DataFrame (for true values)
    - pre_imputation_data: DataFrame with missingness pattern (for mask)
    - eval_data: Evaluation DataFrame (for model performance)
    - true_beta: True coefficients
    - predictor_names: List of predictor names
    - outcome: Response variable
    - col_miss: Columns with missingness
    
    Returns:
    - Dictionary of metrics
    """
    predictors = predictor_names
    target = outcome
    results = []
    
    for full_data in full_list:
        # Imputation RMSE
        rmse = {}
        for col in col_miss:
            mask = pre_imputation_data[col].isna()
            if mask.sum() > 0:
                true_values = original_data.loc[mask, col]
                imputed_values = full_data.loc[mask, col]
                rmse[col] = np.sqrt(mean_squared_error(true_values, imputed_values))
            else:
                rmse[col] = np.nan
        
        # Coefficient metrics
        X = full_data[predictors]
        y = full_data[target]
        if outcome == 'y':
            model = LogisticRegression(random_state=123).fit(X, y)
            coefficients = model.coef_.flatten()
            try:
                cov_matrix = np.linalg.inv(X.T @ X)
                standard_errors = np.sqrt(np.diag(cov_matrix))
            except np.linalg.LinAlgError:
                standard_errors = np.full_like(coefficients, np.nan)
        else:
            model = LinearRegression().fit(X, y)
            coefficients = model.coef_
            try:
                cov_matrix = np.linalg.inv(X.T @ X) * np.var(model.predict(X) - y)
                standard_errors = np.sqrt(np.diag(cov_matrix))
            except np.linalg.LinAlgError:
                standard_errors = np.full_like(coefficients, np.nan)
        
        bias = coefficients - true_beta
        ci_lower = coefficients - 1.96 * standard_errors
        ci_upper = coefficients + 1.96 * standard_errors
        ci_coverage = (true_beta >= ci_lower) & (true_beta <= ci_upper)
        
        # Model performance on eval_data
        X_eval = eval_data[predictors]
        y_eval = eval_data[target]
        if outcome == 'y':
            y_pred = model.predict(X_eval)
            y_pred_proba = model.predict_proba(X_eval)[:, 1]
            accuracy = accuracy_score(y_eval, y_pred)
            auc = roc_auc_score(y_eval, y_pred_proba)
            pred_metric = {'accuracy': accuracy, 'auc': auc}
        else:
            y_pred = model.predict(X_eval)
            rmse_pred = np.sqrt(mean_squared_error(y_eval, y_pred))
            pred_metric = {'rmse_pred': rmse_pred}
        
        results.append({
            'pred_metric': pred_metric,
            'rmse_X1': rmse.get('X1', np.nan),
            'rmse_X2': rmse.get('X2', np.nan),
            'coefficients': coefficients,
            'bias': bias,
            'standard_errors': standard_errors,
            'ci_coverage': ci_coverage.astype(int)
        })
    
    # Pool results
    if len(full_list) > 1:  # MICE
        coefficients = [r['coefficients'] for r in results]
        standard_errors = [r['standard_errors'] for r in results]
        pooled_beta, pooled_se = pool_rubin(coefficients, [se**2 for se in standard_errors], len(full_list))
        pooled_bias = np.mean([r['bias'] for r in results], axis=0)
        pooled_ci_coverage = np.mean([r['ci_coverage'] for r in results], axis=0)
        if outcome == 'y':
            pooled_results = {
                'accuracy': np.mean([r['pred_metric']['accuracy'] for r in results]),
                'auc': np.mean([r['pred_metric']['auc'] for r in results]),
                'rmse_X1': np.nanmean([r['rmse_X1'] for r in results]),
                'rmse_X2': np.nanmean([r['rmse_X2'] for r in results]),
                'bias': pooled_bias,
                'sd': pooled_se,  # Use pooled_se from Rubin's rules
                'ci_coverage': pooled_ci_coverage
                # 'coefficients': pooled_beta
            }
        else:
            pooled_results = {
                'rmse_pred': np.mean([r['pred_metric']['rmse_pred'] for r in results]),
                'rmse_X1': np.nanmean([r['rmse_X1'] for r in results]),
                'rmse_X2': np.nanmean([r['rmse_X2'] for r in results]),
                'bias': pooled_bias,
                'sd': pooled_se,
                'ci_coverage': pooled_ci_coverage
                # 'coefficients': pooled_beta
            }
    else:  # Single/Mean
        pooled_results = {
            'accuracy': results[0]['pred_metric'].get('accuracy', np.nan),
            'auc': results[0]['pred_metric'].get('auc', np.nan),
            'rmse_pred': results[0]['pred_metric'].get('rmse_pred', np.nan),
            'rmse_X1': results[0]['rmse_X1'],
            'rmse_X2': results[0]['rmse_X2'],
            'bias': results[0]['bias'],
            'sd': results[0]['standard_errors'],
            'ci_coverage': results[0]['ci_coverage']
            # 'coefficients': results[0]['coefficients']
        }
    
    return pooled_results

def evaluate_all_imputations(imputed_datasets, original_data, datasets, eval_data, true_beta, predictor_names, col_miss=['X1', 'X2']):
    """
    Evaluate all imputed datasets for matching outcomes.
    
    Parameters:
    - imputed_datasets: Dictionary of imputed datasets
    - original_data: Original complete DataFrame
    - datasets: Dictionary of pre-imputation datasets with missingness
    - eval_data: Evaluation DataFrame
    - true_beta: True coefficients
    - predictor_names: List of predictor names
    - col_miss: Columns with missingness
    
    Returns:
    - DataFrame of results
    """
    results = []
    for dataset_name, methods in imputed_datasets.items():
        pre_imputation_data = datasets[dataset_name]
        for method_name in methods:
            outcomes = ['y'] if 'y' in method_name and 'y_score' not in method_name else ['y_score'] if 'y_score' in method_name else ['y', 'y_score']
            for outcome in outcomes:
                metrics = evaluate_imputation(
                    methods[method_name]['full'],
                    original_data,
                    pre_imputation_data,
                    eval_data,
                    true_beta,
                    predictor_names,
                    outcome=outcome,
                    col_miss=col_miss
                )
                for i, predictor in enumerate(predictor_names):
                    result = {
                        'dataset': dataset_name,
                        'method': method_name,
                        'outcome': outcome,
                        'predictor': predictor,
                        'rmse_X1': metrics['rmse_X1'],
                        'rmse_X2': metrics['rmse_X2'],
                        'bias': metrics['bias'][i],
                        'sd': metrics['sd'][i],
                        'ci_coverage': metrics['ci_coverage'][i]
                        # 'coefficients': metrics.get('coefficients', np.full_like(metrics['bias'], np.nan))[i]
                    }
                    if outcome == 'y':
                        result.update({'accuracy': metrics['accuracy'], 'auc': metrics['auc']})
                    else:
                        result.update({'rmse_pred': metrics['rmse_pred']})
                    results.append(result)
    
    return pd.DataFrame(results)

# Documentation
"""
Functions:
- pool_rubin: Pools coefficients using Rubin's rules
- evaluate_imputation: Evaluates one imputation method
- evaluate_all_imputations: Evaluates imputed datasets for matching outcomes (mean, single_without, mice_without for both y and y_score)
- Description: Computes RMSE, bias, SD, CI coverage, and model performance
"""