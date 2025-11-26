"""
Computational complexity analysis: timing vs n and p.

This script runs simulations with varying n and p values to analyze
computational complexity.
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_simulation import run_simulation

logging.basicConfig(level=logging.WARNING)  # Suppress INFO logs for cleaner output

def analyze_complexity():
    """Run complexity analysis with varying n and p."""
    print("=" * 80)
    print("COMPUTATIONAL COMPLEXITY ANALYSIS")
    print("=" * 80)
    
    # Test parameters
    n_values = [20, 50, 100, 200]  # Sample sizes
    p_values = [5, 10, 20]  # Number of predictors
    num_runs = 1
    
    results = []
    
    print(f"\nTesting {len(n_values)} n values × {len(p_values)} p values = {len(n_values) * len(p_values)} combinations")
    print("This may take a while...\n")
    
    for n in n_values:
        for p in p_values:
            print(f"Testing: n={n}, p={p}...", end=' ', flush=True)
            start_time = time.time()
            try:
                run_simulation(
                    n=[n], p=[p], num_runs=num_runs,
                    continuous_pct=[0.4], integer_pct=[0.4], sparsity=[0.3],
                    include_interactions=[False], include_nonlinear=[False],
                    include_splines=[False], seed=42
                )
                runtime = time.time() - start_time
                results.append({'n': n, 'p': p, 'runtime': runtime, 'success': True})
                print(f"Runtime: {runtime:.2f}s")
            except Exception as e:
                runtime = time.time() - start_time
                results.append({'n': n, 'p': p, 'runtime': runtime, 'success': False, 'error': str(e)})
                print(f"ERROR: {e}")
    
    # Save results
    df = pd.DataFrame(results)
    Path('docs').mkdir(exist_ok=True)
    df.to_csv('docs/complexity_runtime.csv', index=False)
    
    # Generate plots
    print("\nGenerating complexity plots...")
    
    # Plot 1: Runtime vs n (fixing p)
    for p in p_values:
        df_p = df[(df['p'] == p) & df['success']].sort_values('n')
        if len(df_p) > 1:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Linear scale
            ax1.plot(df_p['n'], df_p['runtime'], 'o-', linewidth=2, markersize=8)
            ax1.set_xlabel('Sample Size (n)', fontsize=12)
            ax1.set_ylabel('Runtime (seconds)', fontsize=12)
            ax1.set_title(f'Runtime vs Sample Size (p={p})', fontsize=14)
            ax1.grid(True, alpha=0.3)
            
            # Log-log scale
            ax2.loglog(df_p['n'], df_p['runtime'], 'o-', linewidth=2, markersize=8)
            ax2.set_xlabel('Sample Size (n, log scale)', fontsize=12)
            ax2.set_ylabel('Runtime (seconds, log scale)', fontsize=12)
            ax2.set_title(f'Log-Log: Runtime vs n (p={p})', fontsize=14)
            ax2.grid(True, alpha=0.3)
            
            # Fit trend line
            log_n = np.log(df_p['n'])
            log_runtime = np.log(df_p['runtime'])
            coeffs = np.polyfit(log_n, log_runtime, 1)
            n_fit = np.logspace(np.log10(df_p['n'].min()), np.log10(df_p['n'].max()), 100)
            runtime_fit = np.exp(coeffs[1]) * (n_fit ** coeffs[0])
            ax2.plot(n_fit, runtime_fit, 'r--', linewidth=2, 
                    label=f'Fit: O(n^{coeffs[0]:.2f})')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(f'docs/complexity_n_p{p}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # Plot 2: Runtime vs p (fixing n)
    for n in n_values:
        df_n = df[(df['n'] == n) & df['success']].sort_values('p')
        if len(df_n) > 1:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Linear scale
            ax1.plot(df_n['p'], df_n['runtime'], 'o-', linewidth=2, markersize=8, color='green')
            ax1.set_xlabel('Number of Predictors (p)', fontsize=12)
            ax1.set_ylabel('Runtime (seconds)', fontsize=12)
            ax1.set_title(f'Runtime vs Number of Predictors (n={n})', fontsize=14)
            ax1.grid(True, alpha=0.3)
            
            # Log-log scale
            ax2.loglog(df_n['p'], df_n['runtime'], 'o-', linewidth=2, markersize=8, color='green')
            ax2.set_xlabel('Number of Predictors (p, log scale)', fontsize=12)
            ax2.set_ylabel('Runtime (seconds, log scale)', fontsize=12)
            ax2.set_title(f'Log-Log: Runtime vs p (n={n})', fontsize=14)
            ax2.grid(True, alpha=0.3)
            
            # Fit trend line
            log_p = np.log(df_n['p'])
            log_runtime = np.log(df_n['runtime'])
            coeffs = np.polyfit(log_p, log_runtime, 1)
            p_fit = np.logspace(np.log10(df_n['p'].min()), np.log10(df_n['p'].max()), 100)
            runtime_fit = np.exp(coeffs[1]) * (p_fit ** coeffs[0])
            ax2.plot(p_fit, runtime_fit, 'r--', linewidth=2,
                    label=f'Fit: O(p^{coeffs[0]:.2f})')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(f'docs/complexity_p_n{n}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"\nComplexity analysis complete!")
    print(f"Results saved to docs/complexity_runtime.csv")
    print(f"Plots saved to docs/complexity_*.png")
    print(f"\nSummary:")
    print(f"  Successful runs: {df['success'].sum()}/{len(df)}")
    print(f"  Average runtime: {df[df['success']]['runtime'].mean():.2f}s")
    print(f"  Min runtime: {df[df['success']]['runtime'].min():.2f}s")
    print(f"  Max runtime: {df[df['success']]['runtime'].max():.2f}s")

if __name__ == "__main__":
    analyze_complexity()

