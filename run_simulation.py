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
    param_set, parent_rng = args
    n, p, num_runs, continuous_pct, integer_pct, sparsity, include_interactions, include_nonlinear, include_splines = param_set
    
    # Create parameter suffix for directory and file naming
    param_suffix = (f'n_{n}_p_{p}_runs_{num_runs}_cont_{continuous_pct}_int_{integer_pct}_sparse_{sparsity}_'
                    f'inter_{int(include_interactions)}_nonlin_{int(include_nonlinear)}_splines_{int(include_splines)}')
    
    # Define sub-dirs
    report_dir = f'results/report/{param_suffix}/'
    os.makedirs(report_dir, exist_ok=True)
    
    # Define missingness patterns and imputation methods (UNCHANGED)
    missingness_patterns = [
        MCARPattern(), MARPattern(), MARType2YPattern(), 
        MARType2ScorePattern(), MNARPattern(), MARThresholdPattern()
    ]
    imputation_methods = [
        CompleteData(), MeanImputation(), 
        SingleImputation(use_outcome=None),
        SingleImputation(use_outcome='y'),
        SingleImputation(use_outcome='y_score'),
        MICEImputation(use_outcome=None),
        MICEImputation(use_outcome='y'),
        MICEImputation(use_outcome='y_score'),
        # MissForestImputation(use_outcome=None),
        # MissForestImputation(use_outcome='y'),
        # MissForestImputation(use_outcome='y_score'),
        # MLPImputation(use_outcome=None),
        # MLPImputation(use_outcome='y'),
        # MLPImputation(use_outcome='y_score'),
        # AutoencoderImputation(use_outcome=None),
        # AutoencoderImputation(use_outcome='y'),
        # AutoencoderImputation(use_outcome='y_score'),
        # GAINImputation(use_outcome=None),
        # GAINImputation(use_outcome='y'),
        # GAINImputation(use_outcome='y_score')
    ]
    
    all_run_results = []
    run_rngs = parent_rng.spawn(num_runs)
    for run_idx in range(num_runs):
        # Spawn FRESH RNG for each run
        run_rng = run_rngs[run_idx]
        # Create NEW study for each run with run-specific RNG
        study = SimulationStudy(
            n=n, p=p, num_runs=1,  # CHANGED: num_runs=1 (handled by loop)
            continuous_pct=continuous_pct, integer_pct=integer_pct,
            sparsity=sparsity, include_interactions=include_interactions,
            include_nonlinear=include_nonlinear, include_splines=include_splines,
            rng=run_rng
        )
        
        logger.info(f"Running simulation for param_set: {param_suffix}, run {run_idx}")
        results = study.run_all(missingness_patterns, imputation_methods)
        
        # Aggregate results for THIS RUN (your existing code)
        expected_metrics = [
            # Binary Outcome 'y' Metrics
            'y_log_loss_mean', 'y_log_loss_std',
            
            # Continuous Outcome 'y_score' Metrics
            'y_score_mse_mean', 'y_score_mse_std',
            'y_score_r2_mean', 'y_score_r2_std',
            
            # (The others like 'y_mse_mean', 'y_score_log_loss_mean' are omitted)
        ]
        run_results = []
        for key, result in results.items():
            pattern_name, method_name = key.split(' ')
            # The 'y' column in the results DF now represents the outcome used in the IMPUTATION process (None, 'y', or 'y_score')
            method_instance = next((m for m in imputation_methods if m.name == method_name), None)
            imputation_outcome_used = getattr(method_instance, 'use_outcome', None) if method_instance else None
            result_dict = {
                # result is the single dict from run_scenario, containing all prefixed metrics
                key: result.get(key, np.nan) for key in expected_metrics
            }
            
            result_df = pd.DataFrame([result_dict])
            result_df = result_df.assign(
                missingness=pattern_name, 
                method=method_name, 
                # This 'imputation_outcome_used' column is crucial for analysis
                imputation_outcome_used=imputation_outcome_used or 'none', 
                param_set=param_suffix, 
                run_idx=run_idx  
            )
            run_results.append(result_df)
        
        all_run_results.append(pd.concat(run_results, ignore_index=True))
    
    # Concatenate ALL RUNS
    results_all = pd.concat(all_run_results, ignore_index=True)
    
    return param_set, results_all

def run_simulation(
    n=[50],
    p=[5],        
    num_runs=1,   
    continuous_pct=[0.4], 
    integer_pct=[0.4],
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
    for cont_pct, int_pct in product(continuous_pct, integer_pct):
        if cont_pct + int_pct > 1:
            raise ValueError(f"continuous_pct={cont_pct} + integer_pct={int_pct} > 1. Must leave room for binary covariates.")

    logger.info(f"Starting full factorial simulation with seed={seed}")

    # Generate all parameter combinations
    param_combinations = list(product(n, p, [num_runs], continuous_pct, integer_pct, sparsity,
                                      include_interactions, include_nonlinear, include_splines))
    
    # Prepare arguments for each combination
    parent_rng = default_rng(seed)
    args_list = [(param_set, parent_rng.spawn(1)[0]) for param_set in param_combinations]
    
    # Run in parallel
    with Pool(processes=4) as pool:
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
            'int_pct': float(parts[9]),
            'sparsity': float(parts[11]),
            'interactions': bool(int(parts[13])),
            'nonlinear': bool(int(parts[15])),
            'splines': bool(int(parts[17]))
        }
    
    params_df = results_all['param_set'].apply(parse_param_set).apply(pd.Series)
    results_all = pd.concat([results_all.drop(columns=['param_set']), params_df], axis=1)
    
    # Define the base param for report dir
    param_base = (f'n_{min(n)}_{max(n)}_p_{min(p)}_{max(p)}_runs_{num_runs}_cont_{min(continuous_pct)}_{max(continuous_pct)}_'
                  f'int_{min(integer_pct)}_{max(integer_pct)}_sparse_{min(sparsity)}_{max(sparsity)}_'
                  f'inter_{int(min(include_interactions))}_{int(max(include_interactions))}_'
                  f'nonlin_{int(min(include_nonlinear))}_{int(max(include_nonlinear))}_'
                  f'splines_{int(min(include_splines))}_{int(max(include_splines))}')
    report_dir = f'results/report/{param_base}/'
    os.makedirs(report_dir, exist_ok=True)
    
    results_all.to_csv(os.path.join(report_dir, 'results_all_runs.csv'), index=False)
    logger.info(f"Saved all runs results to {os.path.join(report_dir, 'results_all_runs.csv')}")

    # Aggregate metrics
    metric_cols = [
        'y_log_loss_mean', 'y_log_loss_std', 
        'y_score_mse_mean', 'y_score_mse_std', 
        'y_score_r2_mean', 'y_score_r2_std'
    ]
    results_mean = results_all.groupby([
        'missingness', 'method', 'imputation_outcome_used', 
        'n', 'p', 'cont_pct', 'int_pct',
        'sparsity', 'interactions', 'nonlinear', 'splines'
    ])[metric_cols].mean().reset_index()
    
    results_std_runs = results_all.groupby([
        'missingness', 'method', 'imputation_outcome_used', 
        'n', 'p', 'cont_pct', 'int_pct',
        'sparsity', 'interactions', 'nonlinear', 'splines'
    ])[[m for m in metric_cols if m.endswith('_mean')]].std().reset_index()

    # 3. Rename STD columns for clarity (e.g., y_log_loss_mean -> y_log_loss_mean_std_runs)
    std_col_map = {col: f'{col}_std_runs' for col in results_std_runs.columns if col.endswith('_mean')}
    results_std_runs = results_std_runs.rename(columns=std_col_map)
    
    # 4. Merge mean and std results
    # Use the original mean columns as the merge key
    merge_keys = [
        'missingness', 'method', 'imputation_outcome_used', 
        'n', 'p', 'cont_pct', 'int_pct',
        'sparsity', 'interactions', 'nonlinear', 'splines'
    ]
    
    results_averaged = pd.merge(results_mean, results_std_runs, on=merge_keys, how='left')
    
    # Save the averaged results
    results_averaged.to_csv(os.path.join(report_dir, 'results_averaged.csv'), index=False)
    logger.info(f"Saved averaged results to {os.path.join(report_dir, 'results_averaged.csv')}")

    logger.info(f"Full factorial simulation complete. Results saved in {report_dir}")
    return results_all, results_averaged

if __name__ == "__main__":
    results_all, results_averaged = run_simulation(num_runs=2, n=[50], p=[5], continuous_pct=[0.4], integer_pct=[0.4], sparsity=[0.3],
                                                  include_interactions=[False], include_nonlinear=[False], include_splines=[False])