"""
Runtime and complexity profiling for the simulation study.

This script:
1. Profiles runtime for different parameter values (n, p)
2. Tracks numerical warnings and convergence issues
3. Analyzes computational complexity
4. Generates profiling reports and complexity plots
"""

import cProfile
import pstats
import io
import time
import warnings
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
from collections import defaultdict
import sys

from run_simulation import run_simulation
from numpy.random import default_rng

# Configure logging to capture warnings
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('profiling_warnings.log'),
        logging.StreamHandler()
    ]
)

# Track warnings and errors
warnings_captured = []
errors_captured = []

class WarningTracker:
    """Capture and log all warnings."""
    def __init__(self):
        self.warnings = []
    
    def __call__(self, message, category, filename, lineno, file=None, line=None):
        self.warnings.append({
            'message': str(message),
            'category': category.__name__,
            'filename': filename,
            'lineno': lineno
        })
        return self

# Global warning tracker
warning_tracker = WarningTracker()

def run_profiled_simulation(n, p, num_runs=1, seed=42):
    """
    Run a single simulation with profiling and warning tracking.
    
    Returns:
    -------
    dict : Dictionary with runtime, profile stats, and warnings
    """
    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Profile the simulation
        profiler = cProfile.Profile()
        profiler.enable()
        
        start_time = time.time()
        try:
            results_all, results_avg = run_simulation(
                n=[n], p=[p], num_runs=num_runs,
                continuous_pct=[0.4], integer_pct=[0.4], sparsity=[0.3],
                include_interactions=[False], include_nonlinear=[False],
                include_splines=[False], seed=seed
            )
            success = True
            error_msg = None
        except Exception as e:
            success = False
            error_msg = str(e)
            results_all = None
            results_avg = None
        
        end_time = time.time()
        profiler.disable()
        
        # Get profile stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        profile_output = s.getvalue()
        
        # Collect warnings
        captured_warnings = []
        for warning in w:
            captured_warnings.append({
                'message': str(warning.message),
                'category': warning.category.__name__,
                'filename': warning.filename,
                'lineno': warning.lineno
            })
        
        return {
            'n': n,
            'p': p,
            'num_runs': num_runs,
            'runtime': end_time - start_time,
            'success': success,
            'error': error_msg,
            'warnings': captured_warnings,
            'profile_stats': profile_output,
            'profiler': profiler
        }

def analyze_complexity(results_df):
    """
    Analyze computational complexity from profiling results.
    
    Parameters:
    -----------
    results_df : DataFrame
        DataFrame with columns: n, p, runtime
        
    Returns:
    --------
    dict : Complexity analysis results
    """
    analysis = {}
    
    # Fit complexity for n (fixing p)
    if len(results_df['p'].unique()) == 1:
        p_fixed = results_df['p'].unique()[0]
        df_n = results_df[results_df['p'] == p_fixed].copy()
        if len(df_n) > 1:
            # Log-log regression: log(runtime) ~ log(n)
            df_n['log_n'] = np.log(df_n['n'])
            df_n['log_runtime'] = np.log(df_n['runtime'])
            
            # Fit linear regression
            coeffs = np.polyfit(df_n['log_n'], df_n['log_runtime'], 1)
            slope_n = coeffs[0]
            intercept_n = coeffs[1]
            
            analysis['n_complexity'] = {
                'slope': float(slope_n),
                'intercept': float(intercept_n),
                'complexity': f"O(n^{slope_n:.2f})",
                'r_squared': float(np.corrcoef(df_n['log_n'], df_n['log_runtime'])[0,1]**2)
            }
    
    # Fit complexity for p (fixing n)
    if len(results_df['n'].unique()) == 1:
        n_fixed = results_df['n'].unique()[0]
        df_p = results_df[results_df['n'] == n_fixed].copy()
        if len(df_p) > 1:
            # Log-log regression: log(runtime) ~ log(p)
            df_p['log_p'] = np.log(df_p['p'])
            df_p['log_runtime'] = np.log(df_p['runtime'])
            
            # Fit linear regression
            coeffs = np.polyfit(df_p['log_p'], df_p['log_runtime'], 1)
            slope_p = coeffs[0]
            intercept_p = coeffs[1]
            
            analysis['p_complexity'] = {
                'slope': float(slope_p),
                'intercept': float(intercept_p),
                'complexity': f"O(p^{slope_p:.2f})",
                'r_squared': float(np.corrcoef(df_p['log_p'], df_p['log_runtime'])[0,1]**2)
            }
    
    return analysis

def plot_complexity(results_df, output_dir='docs'):
    """
    Create complexity plots.
    
    Parameters:
    -----------
    results_df : DataFrame
        DataFrame with profiling results
    output_dir : str
        Directory to save plots
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Plot 1: Runtime vs n (fixing p)
    if len(results_df['p'].unique()) == 1:
        p_fixed = results_df['p'].unique()[0]
        df_n = results_df[results_df['p'] == p_fixed].sort_values('n')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Linear scale
        ax1.plot(df_n['n'], df_n['runtime'], 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Sample Size (n)', fontsize=12)
        ax1.set_ylabel('Runtime (seconds)', fontsize=12)
        ax1.set_title(f'Runtime vs Sample Size (p={p_fixed})', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Log-log scale
        ax2.loglog(df_n['n'], df_n['runtime'], 'o-', linewidth=2, markersize=8)
        ax2.set_xlabel('Sample Size (n, log scale)', fontsize=12)
        ax2.set_ylabel('Runtime (seconds, log scale)', fontsize=12)
        ax2.set_title(f'Log-Log Plot: Runtime vs n (p={p_fixed})', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Fit and plot trend line
        if len(df_n) > 1:
            log_n = np.log(df_n['n'])
            log_runtime = np.log(df_n['runtime'])
            coeffs = np.polyfit(log_n, log_runtime, 1)
            n_fit = np.logspace(np.log10(df_n['n'].min()), np.log10(df_n['n'].max()), 100)
            runtime_fit = np.exp(coeffs[1]) * (n_fit ** coeffs[0])
            ax2.plot(n_fit, runtime_fit, 'r--', linewidth=2, 
                    label=f'Fit: O(n^{coeffs[0]:.2f})')
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/complexity_n.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 2: Runtime vs p (fixing n)
    if len(results_df['n'].unique()) == 1:
        n_fixed = results_df['n'].unique()[0]
        df_p = results_df[results_df['n'] == n_fixed].sort_values('p')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Linear scale
        ax1.plot(df_p['p'], df_p['runtime'], 'o-', linewidth=2, markersize=8, color='green')
        ax1.set_xlabel('Number of Predictors (p)', fontsize=12)
        ax1.set_ylabel('Runtime (seconds)', fontsize=12)
        ax1.set_title(f'Runtime vs Number of Predictors (n={n_fixed})', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Log-log scale
        ax2.loglog(df_p['p'], df_p['runtime'], 'o-', linewidth=2, markersize=8, color='green')
        ax2.set_xlabel('Number of Predictors (p, log scale)', fontsize=12)
        ax2.set_ylabel('Runtime (seconds, log scale)', fontsize=12)
        ax2.set_title(f'Log-Log Plot: Runtime vs p (n={n_fixed})', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Fit and plot trend line
        if len(df_p) > 1:
            log_p = np.log(df_p['p'])
            log_runtime = np.log(df_p['runtime'])
            coeffs = np.polyfit(log_p, log_runtime, 1)
            p_fit = np.logspace(np.log10(df_p['p'].min()), np.log10(df_p['p'].max()), 100)
            runtime_fit = np.exp(coeffs[1]) * (p_fit ** coeffs[0])
            ax2.plot(p_fit, runtime_fit, 'r--', linewidth=2,
                    label=f'Fit: O(p^{coeffs[0]:.2f})')
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/complexity_p.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main profiling function."""
    print("=" * 80)
    print("SIMULATION BASELINE PERFORMANCE PROFILING")
    print("=" * 80)
    
    # Test parameters - can be adjusted for faster/slower profiling
    # For full baseline: n_values = [20, 50, 100, 200, 500]
    # For quick test: n_values = [20, 50, 100]
    n_values = [20, 50, 100]  # Sample sizes
    p_values = [5, 10]  # Number of predictors
    num_runs = 1  # Single run for profiling
    
    results = []
    all_warnings = []
    all_errors = []
    
    print(f"\nProfiling {len(n_values)} n values × {len(p_values)} p values = {len(n_values) * len(p_values)} combinations")
    print("This may take a while...\n")
    
    # Profile each combination
    for n in n_values:
        for p in p_values:
            print(f"Profiling: n={n}, p={p}...", end=' ', flush=True)
            result = run_profiled_simulation(n, p, num_runs=num_runs, seed=42)
            results.append(result)
            
            if result['warnings']:
                all_warnings.extend([(n, p, w) for w in result['warnings']])
            
            if not result['success']:
                all_errors.append((n, p, result['error']))
                print(f"ERROR: {result['error']}")
            else:
                print(f"Runtime: {result['runtime']:.2f}s")
    
    # Create results DataFrame
    results_df = pd.DataFrame([
        {
            'n': r['n'],
            'p': r['p'],
            'runtime': r['runtime'],
            'success': r['success'],
            'num_warnings': len(r['warnings'])
        }
        for r in results
    ])
    
    # Analyze complexity
    print("\n" + "=" * 80)
    print("COMPUTATIONAL COMPLEXITY ANALYSIS")
    print("=" * 80)
    
    complexity_analysis = {}
    
    # Analyze n complexity (fixing p)
    for p in p_values:
        df_p = results_df[(results_df['p'] == p) & results_df['success']].copy()
        if len(df_p) > 1:
            analysis = analyze_complexity(df_p)
            if 'n_complexity' in analysis:
                complexity_analysis[f'n_complexity_p{p}'] = analysis['n_complexity']
                print(f"\nFor p={p}:")
                print(f"  Complexity: {analysis['n_complexity']['complexity']}")
                print(f"  R²: {analysis['n_complexity']['r_squared']:.3f}")
    
    # Analyze p complexity (fixing n)
    for n in n_values:
        df_n = results_df[(results_df['n'] == n) & results_df['success']].copy()
        if len(df_n) > 1:
            analysis = analyze_complexity(df_n)
            if 'p_complexity' in analysis:
                complexity_analysis[f'p_complexity_n{n}'] = analysis['p_complexity']
                print(f"\nFor n={n}:")
                print(f"  Complexity: {analysis['p_complexity']['complexity']}")
                print(f"  R²: {analysis['p_complexity']['r_squared']:.3f}")
    
    # Generate plots
    print("\n" + "=" * 80)
    print("GENERATING COMPLEXITY PLOTS")
    print("=" * 80)
    Path('docs').mkdir(exist_ok=True)
    plot_complexity(results_df[results_df['success']], output_dir='docs')
    print("Plots saved to docs/complexity_n.png and docs/complexity_p.png")
    
    # Save detailed results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    # Save runtime data
    results_df.to_csv('docs/profiling_runtime.csv', index=False)
    print("Runtime data saved to docs/profiling_runtime.csv")
    
    # Save warnings summary
    if all_warnings:
        warnings_df = pd.DataFrame([
            {'n': n, 'p': p, 'message': w['message'], 'category': w['category']}
            for n, p, w in all_warnings
        ])
        warnings_df.to_csv('docs/profiling_warnings.csv', index=False)
        print(f"Warnings summary saved to docs/profiling_warnings.csv ({len(all_warnings)} warnings)")
    else:
        print("No warnings captured")
    
    # Save errors
    if all_errors:
        errors_df = pd.DataFrame([
            {'n': n, 'p': p, 'error': err}
            for n, p, err in all_errors
        ])
        errors_df.to_csv('docs/profiling_errors.csv', index=False)
        print(f"Errors saved to docs/profiling_errors.csv ({len(all_errors)} errors)")
    else:
        print("No errors encountered")
    
    # Save complexity analysis
    with open('docs/complexity_analysis.json', 'w') as f:
        json.dump(complexity_analysis, f, indent=2)
    print("Complexity analysis saved to docs/complexity_analysis.json")
    
    # Save detailed profile stats for one example
    if results:
        example_result = results[0]
        with open('docs/profile_stats_example.txt', 'w') as f:
            f.write(f"Profile stats for n={example_result['n']}, p={example_result['p']}\n")
            f.write("=" * 80 + "\n\n")
            f.write(example_result['profile_stats'])
        print("Example profile stats saved to docs/profile_stats_example.txt")
    
    # Print summary
    print("\n" + "=" * 80)
    print("PROFILING SUMMARY")
    print("=" * 80)
    print(f"Total combinations tested: {len(results)}")
    print(f"Successful runs: {results_df['success'].sum()}")
    print(f"Failed runs: {(~results_df['success']).sum()}")
    print(f"Total warnings: {len(all_warnings)}")
    print(f"Total errors: {len(all_errors)}")
    print(f"\nAverage runtime: {results_df[results_df['success']]['runtime'].mean():.2f}s")
    print(f"Min runtime: {results_df[results_df['success']]['runtime'].min():.2f}s")
    print(f"Max runtime: {results_df[results_df['success']]['runtime'].max():.2f}s")
    print("\nProfiling complete! Check docs/ for detailed results.")

if __name__ == "__main__":
    main()

