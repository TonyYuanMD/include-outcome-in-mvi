import os
import numpy as np
import pandas as pd
from generate_data import generate_data
from generate_missingness import define_missingness_patterns, apply_missingness
from impute_stats import impute_datasets
from evaluate_imputations import evaluate_all_imputations

def run_simulation(num_runs=2, n=1000, p=5, continuous_pct=0.4, integer_pct=0.4, sparsity=0.3,
                  include_interactions=False, include_nonlinear=False, include_splines=False,
                  base_seed=123, output_base_dir="syn_data"):
    """
    Run multiple simulations of data generation, missingness, imputation, and evaluation.
    
    Parameters:
    - num_runs: Number of simulation runs
    - n, p, continuous_pct, integer_pct, sparsity, include_interactions, include_nonlinear, include_splines: Data generation parameters
    - base_seed: Base random seed
    - output_base_dir: Base directory for outputs
    
    Returns:
    - List of result DataFrames
    """
    results_all = []
    
    for run in range(1, num_runs + 1):
        run_dir = os.path.join(output_base_dir, f'run_{run}')
        os.makedirs(run_dir, exist_ok=True)
        
        # Generate data
        seed = base_seed + run
        data, covariates, beta = generate_data(n, p, continuous_pct, integer_pct, sparsity,
                                              include_interactions, include_nonlinear, include_splines, seed)
        eval_data, _, _ = generate_data(n, p, continuous_pct, integer_pct, sparsity,
                                        include_interactions, include_nonlinear, include_splines, seed + 1)
        
        data.to_csv(os.path.join(run_dir, 'original_data.csv'), index=False)
        eval_data.to_csv(os.path.join(run_dir, 'eval_data.csv'), index=False)
        np.savetxt(os.path.join(run_dir, 'true_beta.csv'), np.column_stack((['Intercept'] + covariates, beta)), fmt='%s', delimiter=',')
        
        # Apply missingness
        patterns = define_missingness_patterns(data, seed=seed)
        datasets = {}
        for pattern, config in patterns.items():
            datasets[pattern] = apply_missingness(data, config['Mmis'], ['X1', 'X2'], config['vars'], 
                                                os.path.join(run_dir, config['output']))
        
        # Impute datasets
        imputed_datasets = impute_datasets(datasets, seed=seed)
        
        # Evaluate imputations
        true_beta_df = pd.read_csv(os.path.join(run_dir, 'true_beta.csv'), header=None)
        predictor_names = true_beta_df[0][1:].tolist()
        true_beta = true_beta_df[1][1:].astype(float).values
        results = evaluate_all_imputations(imputed_datasets, data, datasets, eval_data, true_beta, predictor_names)
        results['run'] = run
        results.to_csv(os.path.join(run_dir, 'results.csv'), index=False)
        results_all.append(results)
    
    return pd.concat(results_all, ignore_index=True)

# Run simulations
if __name__ == "__main__":
    results = run_simulation()
    print("Simulation Results:")
    print(results.pivot_table(
        index=['run', 'dataset', 'outcome', 'method', 'predictor'],
        values=['rmse_X1', 'rmse_X2', 'bias', 'sd', 'accuracy', 'auc', 'rmse_pred'],
        aggfunc='mean'
    ).reset_index())

# Documentation
"""
Script: run_simulation.py
- Description: Runs multiple simulations of data generation, missingness, imputation, and evaluation
- Outputs: Per-run directories (syn_data/run_i/) with original_data.csv, eval_data.csv, true_beta.csv, missing datasets, and results.csv
"""