"""Evaluation of imputation quality."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, log_loss
import logging

logger = logging.getLogger(__name__)

# Keep evaluate_imputation and evaluate_all_imputations as is, but update evaluate_all_imputations to take dict of {name: list of imputed DFs}
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

# evaluate_imputation remains the same