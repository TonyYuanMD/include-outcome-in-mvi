from multiprocessing import Pool
import os
import json
import logging
from tqdm import tqdm
import pandas as pd
from itertools import product
from pathlib import Path
from functools import lru_cache
from src.pipeline.simulation.data_generators import generate_data
from src.pipeline.simulation.missingness_patterns import (
    MCARPattern, MARPattern, MARType2YPattern, MARType2ScorePattern, MNARPattern, MARThresholdPattern
)
from src.pipeline.simulation.imputation_methods import (
    CompleteData, MeanImputation, SingleImputation, MICEImputation,
    MissForestImputation, MLPImputation, AutoencoderImputation, GAINImputation,
    spawn_rngs
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

def load_config(config_path):
    """
    Load simulation configuration from a JSON file.
    
    Parameters:
    -----------
    config_path : str or Path
        Path to the JSON configuration file
    
    Returns:
    --------
    dict : Configuration dictionary with simulation parameters
    
    Example JSON structure:
    {
        "n": [50, 100],
        "p": [5, 10],
        "num_runs": 2,
        "continuous_pct": [0.4],
        "integer_pct": [0.4],
        "sparsity": [0.3],
        "include_interactions": [false],
        "include_nonlinear": [false],
        "include_splines": [false],
        "seed": 123
    }
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Validate required keys
    required_keys = ['n', 'p', 'num_runs', 'continuous_pct', 'integer_pct', 'sparsity',
                     'include_interactions', 'include_nonlinear', 'include_splines', 'seed']
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")
    
    # Ensure list types for parameters that should be lists
    list_params = ['n', 'p', 'continuous_pct', 'integer_pct', 'sparsity',
                   'include_interactions', 'include_nonlinear', 'include_splines']
    for param in list_params:
        if not isinstance(config[param], list):
            config[param] = [config[param]]
    
    logger.info(f"Loaded configuration from {config_path}")
    return config

def run_single_run(args):
    """Run a single simulation run. Used for parallelization across runs."""
    (n, p, continuous_pct, integer_pct, sparsity, include_interactions, 
     include_nonlinear, include_splines, param_suffix, run_idx, run_rng) = args
    
    # Create missingness patterns and imputation methods inside worker
    # (needed for proper pickling in multiprocessing)
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
    
    # Create method lookup
    method_lookup = {m.name: m for m in imputation_methods}
    
    # Expected metrics
    expected_metrics = [
        'y_log_loss_mean', 'y_log_loss_std',
        'y_score_mse_mean', 'y_score_mse_std',
        'y_score_r2_mean', 'y_score_r2_std',
    ]
    
    # Create NEW study for this run with run-specific RNG
    study = SimulationStudy(
        n=n, p=p, num_runs=1,
        continuous_pct=continuous_pct, integer_pct=integer_pct,
        sparsity=sparsity, include_interactions=include_interactions,
        include_nonlinear=include_nonlinear, include_splines=include_splines,
        rng=run_rng
    )
    
    logger.info(f"Running simulation for param_set: {param_suffix}, run {run_idx}")
    results = study.run_all(missingness_patterns, imputation_methods)
    
    # OPTIMIZATION: Pre-allocate list and use list comprehension where possible
    run_results = []
    for key, result in results.items():
        pattern_name, method_name = key.split(' ', 1)  # Split only on first space
        # OPTIMIZATION: Use dictionary lookup instead of next() iteration
        method_instance = method_lookup.get(method_name)
        imputation_outcome_used = getattr(method_instance, 'use_outcome', None) if method_instance else None
        
        # OPTIMIZATION: Build result_dict more efficiently
        result_dict = {key: result.get(key, np.nan) for key in expected_metrics}
        result_dict.update({
            'missingness': pattern_name,
            'method': method_name,
            'imputation_outcome_used': imputation_outcome_used or 'none',
            'param_set': param_suffix,
            'run_idx': run_idx
        })
        
        run_results.append(pd.DataFrame([result_dict]))
    
    # Return concatenated results for this run
    return pd.concat(run_results, ignore_index=True)

def run_single_combination(args):
    param_set, parent_rng = args
    n, p, num_runs, continuous_pct, integer_pct, sparsity, include_interactions, include_nonlinear, include_splines = param_set
    
    # Create parameter suffix for file naming (used in param_set column)
    param_suffix = (f'n_{n}_p_{p}_runs_{num_runs}_cont_{continuous_pct}_int_{integer_pct}_sparse_{sparsity}_'
                    f'inter_{int(include_interactions)}_nonlin_{int(include_nonlinear)}_splines_{int(include_splines)}')
    
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
    
    # OPTIMIZATION: Pre-define expected_metrics and method lookup outside loop
    expected_metrics = [
        # Binary Outcome 'y' Metrics
        'y_log_loss_mean', 'y_log_loss_std',
        
        # Continuous Outcome 'y_score' Metrics
        'y_score_mse_mean', 'y_score_mse_std',
        'y_score_r2_mean', 'y_score_r2_std',
    ]
    
    # OPTIMIZATION: Create method lookup dictionary once (eliminates repeated searches)
    method_lookup = {m.name: m for m in imputation_methods}
    
    # Prepare RNGs for all runs
    run_rngs = spawn_rngs(parent_rng, num_runs)
    
    # PARALLELIZE ACROSS RUNS with smart limiting to reduce resource contention
    # Each run processes 114 scenarios (6 patterns × 19 methods) sequentially
    # Too many parallel runs → GPU contention, memory pressure, resource competition
    import os
    num_cores_available = int(os.environ.get('SLURM_CPUS_PER_TASK', os.environ.get('NUM_PROCESSES', min(os.cpu_count() or 4, 4))))
    
    # Allow override via environment variable
    max_parallel_runs = int(os.environ.get('MAX_PARALLEL_RUNS', 0))  # 0 = auto
    
    if max_parallel_runs > 0:
        # User-specified limit
        num_cores_for_runs = min(max_parallel_runs, num_cores_available, num_runs)
        logger.info(f"Using MAX_PARALLEL_RUNS={max_parallel_runs} (user-specified)")
    elif num_runs <= 10:
        # Few runs: use all CPUs (works well, no contention)
        num_cores_for_runs = min(num_cores_available, num_runs)
    elif num_runs <= 50:
        # Moderate runs: limit to reduce contention
        num_cores_for_runs = min(16, num_cores_available, num_runs)
    else:
        # Many runs (50+): aggressively limit to 4-8 parallel runs
        # This prevents GPU/memory contention while maintaining some parallelism
        num_cores_for_runs = min(8, num_cores_available, num_runs)
    
    logger.info(f"Parallelizing {num_runs} runs across {num_cores_for_runs} processes for param_set: {param_suffix}")
    if num_runs > 10 and num_cores_for_runs < num_cores_available:
        logger.info(f"  (Limited to {num_cores_for_runs} parallel runs to reduce GPU/memory contention)")
        logger.info(f"  (Set MAX_PARALLEL_RUNS env var to override)")
    
    # Prepare arguments for each run
    run_args_list = [
        (n, p, continuous_pct, integer_pct, sparsity, include_interactions,
         include_nonlinear, include_splines, param_suffix, run_idx, run_rngs[run_idx])
        for run_idx in range(num_runs)
    ]
    
    # Run runs in parallel
    with Pool(processes=num_cores_for_runs) as pool:
        all_run_results = list(tqdm(
            pool.imap(run_single_run, run_args_list), 
            total=num_runs, 
            desc=f"Runs for {param_suffix}"
        ))
    
    # Concatenate ALL RUNS
    results_all = pd.concat(all_run_results, ignore_index=True)
    
    return param_set, results_all

def run_simulation(
    config_file=None,
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
    
    Parameters can be provided either via a JSON config file or directly as function arguments.
    If config_file is provided, it takes precedence over direct arguments.
    
    Parameters:
    -----------
    config_file : str or Path, optional
        Path to JSON configuration file. If provided, other parameters are ignored.
    n : list, default=[50]
        List of number of observations to test
    p : list, default=[5]
        List of number of predictors to test
    num_runs : int, default=1
        Number of simulation runs per parameter combination
    continuous_pct : list, default=[0.4]
        List of proportions of continuous predictors
    integer_pct : list, default=[0.4]
        List of proportions of integer predictors
    sparsity : list, default=[0.3]
        List of sparsity levels
    include_interactions : list, default=[False]
        List of boolean flags for interactions
    include_nonlinear : list, default=[False]
        List of boolean flags for nonlinear terms
    include_splines : list, default=[False]
        List of boolean flags for splines
    seed : int, default=123
        Random seed
    
    Returns:
    --------
    results_all : DataFrame
        DataFrame with all results from all runs
    results_averaged : DataFrame
        DataFrame with averaged metrics across runs
    
    Example:
    --------
    # Using JSON config file
    results_all, results_avg = run_simulation(config_file='config.json')
    
    # Using direct parameters
    results_all, results_avg = run_simulation(n=[50, 100], p=[5], num_runs=2)
    """
    # Load configuration from JSON file if provided
    if config_file is not None:
        config = load_config(config_file)
        n = config['n']
        p = config['p']
        num_runs = config['num_runs']
        continuous_pct = config['continuous_pct']
        integer_pct = config['integer_pct']
        sparsity = config['sparsity']
        include_interactions = config['include_interactions']
        include_nonlinear = config['include_nonlinear']
        include_splines = config['include_splines']
        seed = config['seed']
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
    args_list = [(param_set, spawn_rngs(parent_rng, 1)[0]) for param_set in param_combinations]
    
    # Run in parallel
    # Use SLURM_CPUS_PER_TASK if on HPC, otherwise use cpu_count() or default to 4
    import os
    num_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', os.environ.get('NUM_PROCESSES', min(os.cpu_count() or 4, 4))))
    
    # If only one combination, run it directly (avoids nested multiprocessing)
    # The runs within will be parallelized by run_single_combination
    if len(args_list) == 1:
        logger.info(f"Single parameter combination detected. Parallelizing runs across {num_cores} processes.")
        run_results = [run_single_combination(args_list[0])]
    else:
        # Multiple combinations: parallelize across combinations
        # Note: Each combination will parallelize its runs, which may cause nested multiprocessing
        # For best performance with many runs, consider using only one combination at a time
        logger.info(f"Using {num_cores} parallel processes for {len(args_list)} parameter combinations")
        with Pool(processes=num_cores) as pool:
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

    # OPTIMIZATION: Define groupby keys once and reuse
    groupby_keys = [
        'missingness', 'method', 'imputation_outcome_used', 
        'n', 'p', 'cont_pct', 'int_pct',
        'sparsity', 'interactions', 'nonlinear', 'splines'
    ]
    
    # Aggregate metrics
    metric_cols = [
        'y_log_loss_mean', 'y_log_loss_std', 
        'y_score_mse_mean', 'y_score_mse_std', 
        'y_score_r2_mean', 'y_score_r2_std'
    ]
    
    # OPTIMIZATION: Use agg() with dict for multiple operations in single pass
    # This is more efficient than separate groupby calls
    mean_cols = [m for m in metric_cols if m.endswith('_mean')]
    std_cols = [m for m in metric_cols if m.endswith('_std')]
    
    # Single groupby with aggregation dictionary (more efficient)
    agg_dict = {col: 'mean' for col in metric_cols}
    results_mean = results_all.groupby(groupby_keys, sort=False)[metric_cols].agg(agg_dict).reset_index()
    
    # Calculate std of MEAN metrics across runs (simulation uncertainty of performance)
    results_std_runs = results_all.groupby(groupby_keys, sort=False)[mean_cols].std(numeric_only=True).reset_index()
    
    # Calculate std of STD metrics across runs (variability of imputation uncertainty)
    results_std_std_runs = results_all.groupby(groupby_keys, sort=False)[std_cols].std(numeric_only=True).reset_index()

    # 3. Rename STD columns for clarity
    # Rename: y_log_loss_mean -> y_log_loss_mean_std_runs (simulation uncertainty of performance)
    std_col_map = {col: f'{col}_std_runs' for col in results_std_runs.columns if col.endswith('_mean')}
    results_std_runs = results_std_runs.rename(columns=std_col_map)
    
    # Rename: y_log_loss_std -> y_log_loss_std_std_runs (variability of imputation uncertainty)
    std_std_col_map = {col: f'{col}_std_runs' for col in results_std_std_runs.columns if col.endswith('_std')}
    results_std_std_runs = results_std_std_runs.rename(columns=std_std_col_map)
    
    # 4. Merge mean, std of means, and std of stds
    # OPTIMIZATION: Use reduce() for multiple merges or chain merges more efficiently
    # Since we have the same keys, we can merge in one step using pd.concat after setting index
    # But merge is clearer here, so we'll keep it but ensure we're using the same keys variable
    
    # Merge all three: means, std of means, and std of stds
    # OPTIMIZATION: Use suffixes to avoid column name conflicts, merge in order
    results_averaged = pd.merge(results_mean, results_std_runs, on=groupby_keys, how='left', suffixes=('', '_std_runs'))
    results_averaged = pd.merge(results_averaged, results_std_std_runs, on=groupby_keys, how='left', suffixes=('', '_std_std_runs'))
    
    # Save the averaged results
    results_averaged.to_csv(os.path.join(report_dir, 'results_averaged.csv'), index=False)
    logger.info(f"Saved averaged results to {os.path.join(report_dir, 'results_averaged.csv')}")

    logger.info(f"Full factorial simulation complete. Results saved in {report_dir}")
    return results_all, results_averaged

if __name__ == "__main__":
    results_all, results_averaged = run_simulation(num_runs=2, n=[50], p=[5], continuous_pct=[0.4], integer_pct=[0.4], sparsity=[0.3],
                                                  include_interactions=[False], include_nonlinear=[False], include_splines=[False])