from multiprocessing import Pool
import os
import logging
from tqdm import tqdm
import pandas as pd
from itertools import product
from src.pipeline.simulation.data_generators import generate_data
from src.pipeline.simulation.missingness_patterns import (
    MCARPattern, MARPattern, MARType2YPattern, MARType2ScorePattern, MNARPattern, MARThresholdPattern
)
from src.pipeline.simulation.imputation_methods import (
    CompleteData, MeanImputation, SingleImputation, MICEImputation,
    MissForestImputation, MLPImputation, AutoencoderImputation, GAINImputation
)
from src.pipeline.simulation.evaluator import evaluate_all_imputations
from src.pipeline.simulation.simulator import SimulationStudy
from numpy.random import default_rng
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation.log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

def run_single_combination(args):
    param_set, seed = args
    n, p, num_runs, continuous_pct, sparsity, include_interactions, include_nonlinear, include_splines = param_set
    
    # Create parameter suffix for directory and file naming
    param_suffix = f'n_{n}_p_{p}_runs_{num_runs}_cont_{continuous_pct}_sparse_{sparsity}'
    
    # Define sub-dirs
    report_dir = f'results/report/{param_suffix}/'
    os.makedirs(report_dir, exist_ok=True)
    
    # Initialize SimulationStudy for this parameter set
    study = SimulationStudy(n=n, p=p, num_runs=num_runs, continuous_pct=continuous_pct, sparsity=sparsity,
                            include_interactions=include_interactions, include_nonlinear=include_nonlinear,
                            include_splines=include_splines, seed=seed)
    
    # Define missingness patterns and imputation methods
    missingness_patterns = [
        MCARPattern(),
        MARPattern(),
        MARType2YPattern(),
        MARType2ScorePattern(),
        MNARPattern(),
        MARThresholdPattern()
    ]
    
    imputation_methods = [
        CompleteData(),
        MeanImputation(),
        SingleImputation(use_outcome=None),
        SingleImputation(use_outcome='y'),
        SingleImputation(use_outcome='y_score'),
        MICEImputation(use_outcome=None),
        MICEImputation(use_outcome='y'),
        MICEImputation(use_outcome='y_score'),
        MissForestImputation(use_outcome=None),
        MissForestImputation(use_outcome='y'),
        MissForestImputation(use_outcome='y_score'),
        MLPImputation(use_outcome=None),
        MLPImputation(use_outcome='y'),
        MLPImputation(use_outcome='y_score'),
        AutoencoderImputation(use_outcome=None),
        AutoencoderImputation(use_outcome='y'),
        AutoencoderImputation(use_outcome='y_score'),
        GAINImputation(use_outcome=None),
        GAINImputation(use_outcome='y'),
        GAINImputation(use_outcome='y_score')
    ]
    
    # Run all combinations of patterns and methods
    logger.info(f"Running simulation for param_set: {param_suffix}")
    results = study.run_all(missingness_patterns, imputation_methods)
    
    # Aggregate results with consistent columns
    expected_metrics = ['mse_mean', 'mse_std', 'r2_mean', 'r2_std', 'log_loss_mean', 'log_loss_std']
    all_results = []
    for key, result in results.items():
        # Extract pattern and method from string key
        pattern_name, method_name = key.split(' ')
        # Get the use_outcome from the method (if applicable)
        method_instance = next((m for m in imputation_methods if m.name == method_name), None)
        outcome = getattr(method_instance, 'use_outcome', None) if method_instance else None
        result_dict = {key: result.get(key, np.nan) for key in expected_metrics}
        result_df = pd.DataFrame([result_dict])
        result_df = result_df.assign(missingness=pattern_name, method=method_name, y=outcome or 'none', param_set=param_suffix)
        all_results.append(result_df)
    
    results_all = pd.concat(all_results, ignore_index=True)
    
    return param_set, results_all

def run_simulation(
    n=[50],
    p=[5],        
    num_runs=1,   
    continuous_pct=[0.4],  
    sparsity=[0.3],        
    include_interactions=[False],  
    include_nonlinear=[False],     
    include_splines=[False],       
    seed=123
):
    """
    Run simulation with full factorial design using the refactored framework.
    
    Parameters:
    - n: List of number of observations to test
    - p: List of number of predictors to test
    - num_runs: Number of simulation runs per parameter combination
    - continuous_pct: List of proportions of continuous predictors
    - sparsity: List of sparsity levels
    - include_interactions: List of boolean flags for interactions
    - include_nonlinear: List of boolean flags for nonlinear terms
    - include_splines: List of boolean flags for splines
    - seed: Random seed
    
    Returns:
    - results_all: DataFrame with all results
    - results_averaged: DataFrame with averaged metrics
    """
    # Validate integer_pct for all combinations
    for cont_pct, sparse in product(continuous_pct, sparsity):
        integer_pct = 1 - cont_pct - sparse
        if integer_pct < 0:
            raise ValueError(f"integer_pct={integer_pct} is negative for continuous_pct={cont_pct} and sparsity={sparse}. Sum must be <= 1.")

    logger.info(f"Starting full factorial simulation with seed={seed}")

    # Generate all parameter combinations
    param_combinations = list(product(n, p, [num_runs], continuous_pct, sparsity,
                                      include_interactions, include_nonlinear, include_splines))
    
    # Prepare arguments for each combination
    args_list = [(param_set, seed + i) for i, param_set in enumerate(param_combinations)]
    
    # Run in parallel
    with Pool() as pool:
        run_results = list(tqdm(pool.imap(run_single_combination, args_list), total=len(args_list), desc="Parameter Combinations"))

    # Collect and save results
    all_results = []
    for param_set, results_df in run_results:
        all_results.append(results_df)
    
    results_all = pd.concat(all_results, ignore_index=True)
    
    # Parse param_set into individual columns
    def parse_param_set(param_suffix):
        parts = param_suffix.split('_')
        return {
            'n': int(parts[1]),
            'p': int(parts[3]),
            'runs': int(parts[5]),
            'cont_pct': float(parts[7]),
            'sparsity': float(parts[9])
        }
    
    params_df = results_all['param_set'].apply(parse_param_set).apply(pd.Series)
    results_all = pd.concat([results_all.drop(columns=['param_set']), params_df], axis=1)
    
    # Define the base param for report dir
    param_base = f'n_{min(n)}_{max(n)}_p_{min(p)}_{max(p)}_runs_{num_runs}_cont_{min(continuous_pct)}_{max(continuous_pct)}_sparse_{min(sparsity)}_{max(sparsity)}'
    report_dir = f'results/report/{param_base}/'
    os.makedirs(report_dir, exist_ok=True)
    
    results_all.to_csv(os.path.join(report_dir, 'results_all_runs.csv'), index=False)
    logger.info(f"Saved all runs results to {os.path.join(report_dir, 'results_all_runs.csv')}")

    # Aggregate metrics
    metric_cols = ['mse_mean', 'r2_mean', 'log_loss_mean']  # Adjust based on evaluator output
    results_averaged = results_all.groupby(['missingness', 'method', 'y', 'n', 'p', 'cont_pct', 'sparsity'])[metric_cols].mean().reset_index()
    results_averaged.to_csv(os.path.join(report_dir, 'results_averaged.csv'), index=False)
    logger.info(f"Saved averaged results to {os.path.join(report_dir, 'results_averaged.csv')}")

    logger.info(f"Full factorial simulation complete. Results saved in {report_dir}")
    return results_all, results_averaged

if __name__ == "__main__":
    results_all, results_averaged = run_simulation(num_runs=1, n=[50], p=[5], continuous_pct=[0.4], sparsity=[0.3],
                                                  include_interactions=[False], include_nonlinear=[False], include_splines=[False])  # Simple default case