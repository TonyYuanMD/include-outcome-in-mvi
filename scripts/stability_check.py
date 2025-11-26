"""
Check for numerical warnings and convergence issues across simulation conditions.

This script runs simulations with various parameter combinations and captures
all warnings, errors, and convergence issues.
"""

import warnings
import logging
import pandas as pd
from pathlib import Path
import sys
import os
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_simulation import run_simulation

# Configure logging to capture all warnings
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stability_check.log'),
        logging.StreamHandler()
    ]
)

def check_stability():
    """Run stability checks across various parameter combinations."""
    print("=" * 80)
    print("NUMERICAL STABILITY CHECK")
    print("=" * 80)
    
    # Test various parameter combinations
    test_combinations = [
        {'n': [20], 'p': [5], 'num_runs': 1},
        {'n': [50], 'p': [5], 'num_runs': 1},
        {'n': [100], 'p': [10], 'num_runs': 1},
        {'n': [20], 'p': [10], 'num_runs': 1},
    ]
    
    all_warnings = []
    all_errors = []
    convergence_issues = []
    
    print(f"\nTesting {len(test_combinations)} parameter combinations...")
    print("Capturing warnings and errors...\n")
    
    for i, params in enumerate(test_combinations, 1):
        print(f"Test {i}/{len(test_combinations)}: n={params['n'][0]}, p={params['p'][0]}...", end=' ', flush=True)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            try:
                run_simulation(
                    n=params['n'], p=params['p'], num_runs=params['num_runs'],
                    continuous_pct=[0.4], integer_pct=[0.4], sparsity=[0.3],
                    include_interactions=[False], include_nonlinear=[False],
                    include_splines=[False], seed=42 + i
                )
                
                # Collect warnings
                for warning in w:
                    all_warnings.append({
                        'test': i,
                        'n': params['n'][0],
                        'p': params['p'][0],
                        'category': warning.category.__name__,
                        'message': str(warning.message),
                        'filename': warning.filename,
                        'lineno': warning.lineno
                    })
                
                # Check for convergence-related warnings
                convergence_keywords = ['converge', 'iteration', 'max_iter', 'tol', 'precision']
                for warning in w:
                    msg_lower = str(warning.message).lower()
                    if any(keyword in msg_lower for keyword in convergence_keywords):
                        convergence_issues.append({
                            'test': i,
                            'n': params['n'][0],
                            'p': params['p'][0],
                            'message': str(warning.message)
                        })
                
                print("✓ Success")
                
            except Exception as e:
                all_errors.append({
                    'test': i,
                    'n': params['n'][0],
                    'p': params['p'][0],
                    'error': str(e),
                    'error_type': type(e).__name__
                })
                print(f"✗ ERROR: {e}")
    
    # Save results
    Path('docs').mkdir(exist_ok=True)
    
    if all_warnings:
        warnings_df = pd.DataFrame(all_warnings)
        warnings_df.to_csv('docs/stability_warnings.csv', index=False)
        print(f"\n✓ Captured {len(all_warnings)} warnings")
        print(f"  Saved to docs/stability_warnings.csv")
        
        # Summary by category
        print("\n  Warning summary by category:")
        for category, count in warnings_df['category'].value_counts().items():
            print(f"    {category}: {count}")
    else:
        print("\n✓ No warnings captured")
    
    if all_errors:
        errors_df = pd.DataFrame(all_errors)
        errors_df.to_csv('docs/stability_errors.csv', index=False)
        print(f"\n✗ Encountered {len(all_errors)} errors")
        print(f"  Saved to docs/stability_errors.csv")
    else:
        print("\n✓ No errors encountered")
    
    if convergence_issues:
        convergence_df = pd.DataFrame(convergence_issues)
        convergence_df.to_csv('docs/stability_convergence.csv', index=False)
        print(f"\n⚠ Found {len(convergence_issues)} convergence-related warnings")
        print(f"  Saved to docs/stability_convergence.csv")
    else:
        print("\n✓ No convergence issues detected")
    
    print("\n" + "=" * 80)
    print("STABILITY CHECK SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(test_combinations)}")
    print(f"Warnings: {len(all_warnings)}")
    print(f"Errors: {len(all_errors)}")
    print(f"Convergence issues: {len(convergence_issues)}")
    print("\nStability check complete!")
    print("Check docs/stability_*.csv for detailed results.")

if __name__ == "__main__":
    check_stability()

