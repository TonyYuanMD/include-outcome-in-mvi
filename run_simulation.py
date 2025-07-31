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
    - output_base_dir: Base directory for outputs (will be modified to include parameters)
    
    Returns:
    - DataFrame of all results
    """
    # Create parameter-specific output directory
    param_dir = f'n_{n}_p_{p}_runs_{num_runs}'
    output_base_dir = os.path.join(output_base_dir, param_dir)
    os.makedirs(output_base_dir, exist_ok=True)
    
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
    
    # Save combined results
    all_results = pd.concat(results_all, ignore_index=True)
    all_results.to_csv(os.path.join(output_base_dir, 'results_all_runs.csv'), index=False)
    
    # Save metadata
    metadata = pd.DataFrame([{
        'num_runs': num_runs,
        'n': n,
        'p': p,
        'continuous_pct': continuous_pct,
        'integer_pct': integer_pct,
        'sparsity': sparsity,
        'include_interactions': include_interactions,
        'include_nonlinear': include_nonlinear,
        'include_splines': include_splines,
        'base_seed': base_seed,
        'output_dir': output_base_dir
    }])
    metadata_file = os.path.join('syn_data', 'metadata.csv')
    if os.path.exists(metadata_file):
        existing_metadata = pd.read_csv(metadata_file)
        metadata = pd.concat([existing_metadata, metadata], ignore_index=True)
    metadata.to_csv(metadata_file, index=False)
    
    return all_results

# Run simulations
if __name__ == "__main__":
    results = run_simulation()
    print("Simulation Results (Per Run):")
    print(results.pivot_table(
        index=['run', 'dataset', 'outcome', 'method', 'predictor'],
        values=['rmse_X1', 'rmse_X2', 'bias', 'sd', 'accuracy', 'auc', 'rmse_pred'],
        aggfunc='mean'
    ).reset_index())
    
    print("\nAveraged Results Across Runs:")
    averaged_results = results.pivot_table(
        index=['dataset', 'outcome', 'method', 'predictor'],
        values=['rmse_X1', 'rmse_X2', 'bias', 'sd', 'accuracy', 'auc', 'rmse_pred'],
        aggfunc='mean'
    ).reset_index()
    print(averaged_results)
    averaged_results.to_csv(os.path.join('syn_data', 'n_1000_p_5_runs_2', 'results_averaged.csv'), index=False)

# Documentation
"""
Script: run_simulation.py
- Description: Runs multiple simulations of data generation, missingness, imputation, and evaluation
- Outputs: 
  - Per-run directories (syn_data/n_{n}_p_{p}_runs_{num_runs}/run_i/) with original_data.csv, eval_data.csv, true_beta.csv, missing datasets, and results.csv
  - Combined results (syn_data/n_{n}_p_{p}_runs_{num_runs}/results_all_runs.csv)
  - Averaged results (syn_data/n_{n}_p_{p}_runs_{num_runs}/results_averaged.csv)
  - Metadata (syn_data/metadata.csv) with simulation parameters
"""