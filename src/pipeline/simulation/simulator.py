"""Simulation study orchestration."""

import os
import logging
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool
from functools import partial
from src.pipeline.simulation.data_generators import generate_data
from src.pipeline.simulation.evaluator import evaluate_all_imputations, evaluate_imputation
from numpy.random import default_rng

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class SimulationStudy:
    def __init__(self, n=1000, p=5, num_runs=2, continuous_pct=0.4, integer_pct=0.4, sparsity=0.3, 
                 include_interactions=False, include_nonlinear=False, include_splines=False, rng=None, seed=None):
        if rng is not None:
            self.rng = rng
        else:
            self.rng = default_rng(seed)
        self.n = n
        self.p = p
        self.num_runs = num_runs
        self.continuous_pct = continuous_pct
        self.integer_pct = integer_pct
        self.sparsity = sparsity
        self.include_interactions = include_interactions
        self.include_nonlinear = include_nonlinear
        self.include_splines = include_splines
        self.seed = seed
        self.output_dir = f'results/report/n_{n}_p_{p}_runs_{num_runs}_cont_{continuous_pct}_int_{integer_pct}_sparse_{sparsity}/'
        os.makedirs(self.output_dir, exist_ok=True)

        if self.continuous_pct + self.integer_pct > 1:
            raise ValueError(f"continuous_pct + integer_pct must be <= 1. Got {self.continuous_pct} + {self.integer_pct} = {self.continuous_pct + self.integer_pct}.")
        if not (0 <= self.sparsity <= 1):
            raise ValueError(f"sparsity must be between 0 and 1. Got {self.sparsity}.")
    
    def run_scenario(self, missingness_pattern, imputation_method, rng):
        """
        Runs one scenario: generates train/test data, applies missingness to train, 
        imputes, and evaluates utility on the test set.
        """
        # We generate two datasets to serve as separate, complete TRUE train and test sets.
        
        # 1. Generate TRUE Training Data (for missingness application)
        # Use a spawn of the scenario RNG for data generation
        train_data_rng = rng.spawn(1)[0]
        train_data_true, _, _ = generate_data(
            self.n, self.p, self.continuous_pct, self.integer_pct, self.sparsity,
            self.include_interactions, self.include_nonlinear, self.include_splines,
            seed=train_data_rng.integers(0, 2**32)
        )
        
        # 2. Generate TRUE Test Data (for evaluation)
        # Use another spawn for the test set. Note: Test set size is also 'n'.
        test_data_rng = rng.spawn(1)[0]
        test_data_true, _, _ = generate_data(
            self.n, self.p, self.continuous_pct, self.integer_pct, self.sparsity,
            self.include_interactions, self.include_nonlinear, self.include_splines,
            seed=test_data_rng.integers(0, 2**32)
        )
        
        # 3. Apply missingness to TRAINING data
        dat_miss = missingness_pattern.apply(train_data_true, seed=rng.integers(0, 2**32))
        
        # 4. Impute TRAINING data (list of imputed datasets)
        imputed_list = imputation_method.impute(dat_miss, train_data_true, n_imputations=5, seed=rng.integers(0, 2**32))
        
        # 5. Evaluate utility using TEST data
        # Evaluate performance for the binary outcome 'y'
        metrics_y = evaluate_imputation(imputed_list, test_data_true, y='y')
        
        # Evaluate performance for the continuous outcome 'y_score'
        metrics_score = evaluate_imputation(imputed_list, test_data_true, y='y_score')

        # Combine metrics into a single dictionary
        results = {
            'y_log_loss_mean': metrics_y.get('log_loss_mean'),
            'y_log_loss_std': metrics_y.get('log_loss_std'),
            'y_mse_mean': metrics_y.get('mse_mean'), # MSE for y will be NaN/0 from linear regression, but kept for completeness
            'y_mse_std': metrics_y.get('mse_std'),
            'y_r2_mean': metrics_y.get('r2_mean'),
            'y_r2_std': metrics_y.get('r2_std'),
            
            'y_score_mse_mean': metrics_score.get('mse_mean'),
            'y_score_mse_std': metrics_score.get('mse_std'),
            'y_score_r2_mean': metrics_score.get('r2_mean'),
            'y_score_r2_std': metrics_score.get('r2_std'),
            'y_score_log_loss_mean': metrics_score.get('log_loss_mean'), # Log loss for y_score will be NaN/0 from log loss, but kept for completeness
            'y_score_log_loss_std': metrics_score.get('log_loss_std'),
        }
            
        return results
    
    def run_all(self, missingness_patterns, imputation_methods):
        results = {}
        
        # We spawn a single RNG to manage the sequence of scenarios
        scenario_rng = self.rng.spawn(1)[0] 
        
        for pattern in missingness_patterns:
            for method in imputation_methods:
                # Use a unique spawn for this scenario run's randomness
                run_rng = scenario_rng.spawn(1)[0]
                
                # The run_scenario function handles its internal data/missingness RNGs
                metrics = self.run_scenario(
                    pattern, method, rng=run_rng
                )
                
                # The metrics are returned with prefixes (y_ and y_score_)
                results[f"{pattern.name} {method.name}"] = metrics
        
        return results
    
    def run_parallel(self):
        # Use multiprocessing for runs
        # ... (implement as in earlier suggestion, using partial and Pool)
        pass  # Expand based on previous code