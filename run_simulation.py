import os
import logging
from tqdm import tqdm
from generate_data import generate_data
from generate_missingness import define_missingness_patterns, apply_missingness
from impute_stats import impute_datasets
from evaluate_imputations import evaluate_all_imputations
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

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
    output_dir = f'syn_data/n_{n}_p_{p}_runs_{num_runs}_cont_{continuous_pct}_sparse_{sparsity}/'
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Starting simulation: n={n}, p={p}, runs={num_runs}, seed={seed}")
    
    for run in tqdm(range(num_runs), desc="Simulation Runs"):
        run_dir = os.path.join(output_dir, f'run_{run}')
        os.makedirs(run_dir, exist_ok=True)
        
        # Generate data
        logger.info(f"Run {run}: Generating data")
        data, covariates, beta = generate_data(
            n=n,
            p=p,
            continuous_pct=continuous_pct,
            integer_pct=integer_pct,
            sparsity=sparsity,
            include_interactions=include_interactions,
            include_nonlinear=include_nonlinear,
            include_splines=include_splines,
            seed=seed + run
        )
        data.to_csv(os.path.join(run_dir, 'complete_data.csv'), index=False)
        
        # Generate missingness patterns
        logger.info(f"Run {run}: Applying missingness")
        patterns = define_missingness_patterns(data, seed=seed + run)
        datasets = {}
        for name, pattern in patterns.items():
            datasets[name] = apply_missingness(
                data,
                pattern['Mmis'],
                col_miss=['X1', 'X2'],
                vars=pattern['vars'],
                output_file=os.path.join(run_dir, pattern['output'])
            )
            logger.info(f"Run {run}: Saved {name} dataset with missingness")
        
        # Impute datasets
        logger.info(f"Run {run}: Performing imputation")
        imputed_datasets = impute_datasets(datasets, col_miss=['X1', 'X2'], seed=seed + run)
        
        # Save imputed datasets
        for dataset_name, methods in imputed_datasets.items():
            for method, imputed in methods.items():
                for idx, df in enumerate(imputed['full']):
                    df.to_csv(os.path.join(run_dir, f'{dataset_name}_{method}_imputed_{idx}.csv'), index=False)
                logger.info(f"Run {run}: Saved {dataset_name} - {method} imputed datasets")
        
        # Evaluate
        logger.info(f"Run {run}: Evaluating imputations")
        results[run] = evaluate_all_imputations(data, imputed_datasets, output_dir=run_dir)
        logger.info(f"Run {run}: Evaluation complete")
    
    # Aggregate results
    logger.info("Aggregating results across runs")
    results_all = pd.concat([results[run]['results_all'] for run in range(num_runs)])
    results_averaged = results_all.groupby(['missingness', 'method', 'y']).mean().reset_index()
    results_all.to_csv(os.path.join(output_dir, 'results_all_runs.csv'), index=False)
    results_averaged.to_csv(os.path.join(output_dir, 'results_averaged.csv'), index=False)
    
    logger.info(f"Simulation complete. Results saved in {output_dir}")
    return results

if __name__ == "__main__":
    results = run_simulation(num_runs=1)  # Default to 1 run for testing

# Documentation
"""
Function:
- run_simulation: Runs simulation with data generation, missingness, imputation, and evaluation.
  Saves intermediate datasets and logs progress.
"""