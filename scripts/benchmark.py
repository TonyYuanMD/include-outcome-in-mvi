"""
Benchmark comparison: baseline vs optimized version.

This script runs the same simulation with both versions to compare performance.
Note: Since we've already optimized, this compares current version performance
across different parameter sets to establish benchmarks.
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_simulation import run_simulation

logging.basicConfig(level=logging.WARNING)  # Suppress INFO logs

def run_benchmark():
    """Run benchmark tests."""
    print("=" * 80)
    print("BENCHMARK: OPTIMIZED VERSION PERFORMANCE")
    print("=" * 80)
    
    # Benchmark parameter sets
    benchmark_sets = [
        {'name': 'Small', 'n': [20], 'p': [5], 'num_runs': 1},
        {'name': 'Medium', 'n': [50], 'p': [5], 'num_runs': 1},
        {'name': 'Large', 'n': [100], 'p': [10], 'num_runs': 1},
    ]
    
    results = []
    
    print(f"\nRunning {len(benchmark_sets)} benchmark tests...\n")
    
    for bench in benchmark_sets:
        name = bench['name']
        n = bench['n'][0]
        p = bench['p'][0]
        
        print(f"Benchmark: {name} (n={n}, p={p})...", end=' ', flush=True)
        
        start_time = time.time()
        try:
            results_all, results_avg = run_simulation(
                n=bench['n'], p=bench['p'], num_runs=bench['num_runs'],
                continuous_pct=[0.4], integer_pct=[0.4], sparsity=[0.3],
                include_interactions=[False], include_nonlinear=[False],
                include_splines=[False], seed=42
            )
            runtime = time.time() - start_time
            
            results.append({
                'name': name,
                'n': n,
                'p': p,
                'num_runs': bench['num_runs'],
                'runtime': runtime,
                'success': True,
                'num_scenarios': len(results_all) if results_all is not None else 0
            })
            print(f"✓ {runtime:.2f}s")
            
        except Exception as e:
            runtime = time.time() - start_time
            results.append({
                'name': name,
                'n': n,
                'p': p,
                'num_runs': bench['num_runs'],
                'runtime': runtime,
                'success': False,
                'error': str(e)
            })
            print(f"✗ ERROR: {e}")
    
    # Save results
    df = pd.DataFrame(results)
    Path('docs').mkdir(exist_ok=True)
    df.to_csv('docs/benchmark_results.csv', index=False)
    
    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    if df['success'].any():
        successful = df[df['success']]
        print(f"\nSuccessful benchmarks: {len(successful)}/{len(df)}")
        print(f"\nRuntime by test:")
        for _, row in successful.iterrows():
            print(f"  {row['name']:8s} (n={row['n']:3d}, p={row['p']:2d}): {row['runtime']:6.2f}s")
        
        print(f"\nOverall statistics:")
        print(f"  Average runtime: {successful['runtime'].mean():.2f}s")
        print(f"  Min runtime: {successful['runtime'].min():.2f}s")
        print(f"  Max runtime: {successful['runtime'].max():.2f}s")
        print(f"  Total runtime: {successful['runtime'].sum():.2f}s")
    
    if not df['success'].all():
        print(f"\nFailed benchmarks: {len(df[~df['success']])}")
        for _, row in df[~df['success']].iterrows():
            print(f"  {row['name']}: {row.get('error', 'Unknown error')}")
    
    print(f"\nResults saved to docs/benchmark_results.csv")
    print("\nBenchmark complete!")

if __name__ == "__main__":
    run_benchmark()

