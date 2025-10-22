import pytest
import pandas as pd
import numpy as np
import os
import shutil
from src.pipeline.simulation.simulator import SimulationStudy
from src.pipeline.simulation.missingness_patterns import MCARPattern
from src.pipeline.simulation.imputation_methods import MeanImputation, MICEImputation
from src.pipeline.simulation.evaluator import evaluate_imputation

# --- Fixture for common setup ---

@pytest.fixture(scope="module")
def common_study_params():
    """Provides standard parameters for a fast simulation study."""
    return {
        'n': 50,
        'p': 5,
        'num_runs': 1,
        'continuous_pct': 0.8,
        'integer_pct': 0.0,
        'sparsity': 0.0,
        'seed': 42
    }

@pytest.fixture(scope="module")
def setup_study(common_study_params):
    """Creates and returns a basic SimulationStudy instance."""
    return SimulationStudy(**common_study_params)

# --- Test Functions ---

# ----------------------------------------------------------------------
# TEST 1: Utility Evaluation Sanity Check (Core Logic)
# Ensures the evaluator selects the correct metrics for binary vs. continuous outcomes.
# ----------------------------------------------------------------------
def test_01_utility_evaluation_metrics_selection(setup_study):
    """Verifies that evaluate_imputation calculates Log Loss for 'y' and MSE/R2 for 'y_score', and omits the others."""
    
    # Generate two separate complete datasets for train/test
    train_data_true, _, _ = setup_study.generate_data(seed=1)
    test_data_true, _, _ = setup_study.generate_data(seed=2)
    
    # Apply missingness and impute (using MeanImputation for simplicity)
    missingness_pattern = MCARPattern()
    imputation_method = MeanImputation()
    dat_miss = missingness_pattern.apply(train_data_true, seed=10)
    imputed_list = imputation_method.impute(dat_miss, train_data_true, seed=20)
    
    # 1. Evaluate Binary Outcome ('y')
    metrics_y = evaluate_imputation(imputed_list, test_data_true, y='y')
    assert 'log_loss_mean' in metrics_y, "Log Loss should be calculated for binary 'y'"
    assert 'mse_mean' not in metrics_y, "MSE should NOT be calculated for binary 'y'" # Will be NaN/None

    # 2. Evaluate Continuous Outcome ('y_score')
    metrics_score = evaluate_imputation(imputed_list, test_data_true, y='y_score')
    assert 'mse_mean' in metrics_score, "MSE should be calculated for continuous 'y_score'"
    assert 'r2_mean' in metrics_score, "R2 should be calculated for continuous 'y_score'"
    assert 'log_loss_mean' not in metrics_score, "Log Loss should NOT be calculated for continuous 'y_score'" # Will be NaN/None

# ----------------------------------------------------------------------
# TEST 2: Multi-Imputation STD Check (Imputation Uncertainty)
# Ensures MICE's inherent randomness is captured by the STD calculation.
# ----------------------------------------------------------------------
def test_02_mice_imputation_uncertainty(setup_study):
    """Verifies that MICE (n_imputations=5) produces a non-zero STD reflecting imputation variability."""
    
    # MICEImputation is a stochastic process
    mice_imputer = MICEImputation(n_imputations=5)
    
    train_data_true, _, _ = setup_study.generate_data(seed=10)
    test_data_true, _, _ = setup_study.generate_data(seed=20)
    
    missingness_pattern = MCARPattern()
    dat_miss = missingness_pattern.apply(train_data_true, seed=30)
    
    # Impute 5 times using the same input data
    imputed_list = mice_imputer.impute(dat_miss, train_data_true, seed=40)
    
    # Evaluate Continuous Outcome ('y_score')
    metrics_score = evaluate_imputation(imputed_list, test_data_true, y='y_score')
    
    # Assert that the STD across the 5 imputations is significantly non-zero (greater than a small epsilon)
    # This verifies that the imputation randomness is being captured.
    assert metrics_score['r2_std'] > 1e-6, "R2 STD across imputations should be > 0 for MICE (Imputation Uncertainty)"
    assert metrics_score['mse_std'] > 1e-6, "MSE STD across imputations should be > 0 for MICE"

# ----------------------------------------------------------------------
# TEST 3: Simulation Uncertainty Check (Robustness/STD Runs)
# Manually verifies that the overall scenario results vary when starting data is regenerated.
# ----------------------------------------------------------------------
def test_03_simulation_uncertainty_across_runs(common_study_params, setup_study):
    """Verifies that running the scenario with different starting data seeds produces a non-zero STD across runs."""
    
    num_test_runs = 5
    mice_imputer = MICEImputation(n_imputations=2) 
    log_loss_means = []
    
    # Manually execute the run_scenario logic multiple times with different base seeds
    for run_idx in range(num_test_runs):
        # Create a unique RNG for the run (simulating run_simulation's loop)
        run_rng = np.random.default_rng(100 + run_idx) 
        
        # The run_scenario logic handles data generation randomness and evaluation
        metrics = setup_study.run_scenario(
            MCARPattern(), mice_imputer, run_rng
        )
        log_loss_means.append(metrics['y_log_loss_mean'])
        
    # Calculate the Standard Deviation of the Log Loss means
    ll_std = np.std(log_loss_means)
    
    # Assert that the variability across the different data generations is significantly non-zero
    assert ll_std > 1e-4, "STD of Log Loss means across multiple runs should be > 0 (Simulation Uncertainty)"