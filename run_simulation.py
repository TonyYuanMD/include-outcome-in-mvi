from multiprocessing import Pool
import os
import logging
from tqdm import tqdm
import pandas as pd
from src.pipeline.generate_data import generate_data
from src.pipeline.generate_missingness import define_missingness_patterns, apply_missingness
from src.pipeline.impute_stats import impute_datasets
from src.pipeline.evaluate_imputations import evaluate_all_imputations

# Configure logging (high-level only)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation.log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

def run_single_simulation(args):
    run, n, p, num_runs, continuous_pct, sparsity, include_interactions, include_nonlinear, include_splines, seed = args
    param_suffix = f'n_{n}_p_{p}_runs_{num_runs}_cont_{continuous_pct}_sparse_{sparsity}'
    run_suffix = f'run_{run}'
    
    # Define sub-dirs (unchanged)
    raw_dir = f'data/raw/{param_suffix}/{run_suffix}/'
    missing_dir = f'artifacts/missing/{param_suffix}/{run_suffix}/'
    imputed_dir = f'artifacts/imputed/{param_suffix}/{run_suffix}/'
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(missing_dir, exist_ok=True)
    os.makedirs(imputed_dir, exist_ok=True)
    
    # Generate data (unchanged)
    logger.info(f"Run {run}: Generating data")
    data, covariates, beta = generate_data(
        n=n, p=p, continuous_pct=continuous_pct, integer_pct=1 - continuous_pct - sparsity,
        sparsity=sparsity, include_interactions=include_interactions,
        include_nonlinear=include_nonlinear, include_splines=include_splines,
        seed=seed + run
    )
    data.to_csv(os.path.join(raw_dir, 'complete_data.csv'), index=False)
    
    # Apply missingness (unchanged, but save here)
    logger.info(f"Run {run}: Applying missingness")
    patterns = define_missingness_patterns(data, seed=seed + run)
    datasets = {}
    for name, pattern in patterns.items():
        dat_miss = apply_missingness(data, pattern['Mmis'], col_miss=['X1', 'X2'], vars=pattern['vars'])
        missing_file = os.path.join(missing_dir, f'{name}_data.csv')
        dat_miss.to_csv(missing_file, index=False)
        datasets[name] = dat_miss
        logger.info(f"Run {run}: Saved {name} dataset with missingness")
    
    # Impute datasets (unchanged)
    logger.info(f"Run {run}: Performing imputation")
    imputed_datasets = {}
    for dataset_name, dataset_data in datasets.items():
        logger.info(f"Run {run}: Imputing {dataset_name}")
        imputed_datasets[dataset_name] = impute_datasets(dataset_data, data, col_miss=['X1', 'X2'], seed=seed + run)
        for method, imputed in imputed_datasets[dataset_name].items():
            for idx, df in enumerate(imputed['full']):
                imputed_file = os.path.join(imputed_dir, f'{dataset_name}_{method}_imputed_{idx}.csv')
                df.to_csv(imputed_file, index=False)
            logger.info(f"Run {run}: Saved {dataset_name} - {method} imputed datasets")
    
    # Evaluate (unchanged)
    logger.info(f"Run {run}: Evaluating imputations")
    eval_result = evaluate_all_imputations(data, imputed_datasets, output_dir=f'results/report/{param_suffix}/')
    logger.info(f"Run {run}: Evaluation complete")
    
    return run, eval_result

def run_simulation(n=1000, p=5, num_runs=2, continuous_pct=0.4, sparsity=0.3, include_interactions=False, include_nonlinear=False, include_splines=False, seed=123):
    """
    Run simulation with data generation, missingness, imputation, and evaluation.
    
    Parameters:
    - n: Number of observations
    - p: Number of predictors
    - num_runs: Number of simulation runs
    - continuous_pct: Proportion of continuous predictors
    - sparsity: Sparsity level for coefficients
    - include_interactions: Include pairwise interaction terms
    - include_nonlinear: Include sin, cos transformations
    - include_splines: Include spline basis expansion
    - seed: Random seed
    
    Returns:
    - results: Dictionary of results
    """
    # Validate integer_pct
    integer_pct = 1 - continuous_pct - sparsity
    if integer_pct < 0:
        raise ValueError(f"integer_pct={integer_pct} is negative. Adjust continuous_pct={continuous_pct} and sparsity={sparsity} so their sum <= 1.")

    results = {}
    param_suffix = f'n_{n}_p_{p}_runs_{num_runs}_cont_{continuous_pct}_sparse_{sparsity}'
    report_dir = f'results/report/{param_suffix}/'
    os.makedirs(report_dir, exist_ok=True)
    
    logger.info(f"Starting simulation: n={n}, p={p}, runs={num_runs}, seed={seed}")
    
    # Prepare arguments for each run
    args_list = [(run, n, p, num_runs, continuous_pct, sparsity, include_interactions, include_nonlinear, include_splines, seed) for run in range(num_runs)]
    
    # Run in parallel
    with Pool() as pool:  # Use all cores; or Pool(processes=4) to limit
        run_results = pool.map(run_single_simulation, args_list)
    
    # Collect results into dict
    results = {run: eval_result for run, eval_result in run_results}
    
    # Aggregate results (unchanged)
    logger.info("Aggregating results across runs")
    results_all = pd.concat([results[run]['results_all'] for run in range(num_runs)])
    results_all.to_csv(os.path.join(report_dir, 'results_all_runs.csv'), index=False)
    results_averaged = results_all.groupby(['missingness', 'method', 'y']).mean().reset_index()
    results_averaged.to_csv(os.path.join(report_dir, 'results_averaged.csv'), index=False)
    
    logger.info(f"Simulation complete. Results saved in {report_dir}")
    return results_all, results_averaged

if __name__ == "__main__":
    results_all, results_averaged = run_simulation(num_runs=20, n=100)  # Default to 1 run for testing