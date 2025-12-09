"""
Demo script for the Include Y in Multivariate Value Imputation (MVI) project.

This script demonstrates basic usage of the simulation framework with a simple example.
It runs a quick simulation to showcase the workflow and output format.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.pipeline.simulation.simulator import SimulationStudy
from src.pipeline.simulation.missingness_patterns import (
    MCARPattern, MARPattern, MNARPattern
)
from src.pipeline.simulation.imputation_methods import (
    MeanImputation, MICEImputation, MissForestImputation
)
from numpy.random import default_rng
import pandas as pd

def demo_basic_simulation():
    """
    Run a basic simulation with a single scenario.
    
    This demonstrates:
    - Creating a simulation study
    - Applying a missingness pattern
    - Imputing with a simple method
    - Evaluating utility metrics
    """
    print("=" * 70)
    print("DEMO: Basic Simulation Scenario")
    print("=" * 70)
    print()
    
    # Create a small simulation study
    print("Creating simulation study with:")
    print("  - Sample size (n): 100")
    print("  - Number of predictors (p): 5")
    print("  - Number of runs: 1")
    print()
    
    study = SimulationStudy(
        n=100,
        p=5,
        num_runs=1,
        continuous_pct=0.4,
        integer_pct=0.4,
        sparsity=0.3,
        include_interactions=False,
        include_nonlinear=False,
        include_splines=False,
        seed=42
    )
    
    # Define missingness pattern and imputation method
    pattern = MCARPattern()
    method = MeanImputation()
    
    print(f"Missingness Pattern: {pattern.name}")
    print(f"Imputation Method: {method.name}")
    print()
    print("Running scenario...")
    print()
    
    # Run the scenario
    rng = default_rng(42)
    results = study.run_scenario(pattern, method, run_rng=rng)
    
    # Display results
    print("Results:")
    print("-" * 70)
    for key, value in results.items():
        if value is not None:
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
    print()
    
    return results

def demo_multiple_methods():
    """
    Compare multiple imputation methods on the same scenario.
    """
    print("=" * 70)
    print("DEMO: Comparing Multiple Imputation Methods")
    print("=" * 70)
    print()
    
    # Create simulation study
    study = SimulationStudy(
        n=100,
        p=10,
        num_runs=100,
        continuous_pct=0.4,
        integer_pct=0.4,
        sparsity=0.3,
        include_interactions=True,
        include_nonlinear=True,
        include_splines=False,
        seed=42
    )
    
    # Test multiple methods
    pattern = MCARPattern()
    methods = [
        MeanImputation(),
        MICEImputation(use_outcome='y', n_imputations=5),
        MissForestImputation(use_outcome='y', n_imputations=5)
    ]
    
    print(f"Missingness Pattern: {pattern.name}")
    print(f"Testing {len(methods)} imputation methods:")
    print()
    
    results_comparison = {}
    
    for method in methods:
        print(f"  Running {method.name}...")
        rng = default_rng(42)
        results = study.run_scenario(pattern, method, run_rng=rng)
        results_comparison[method.name] = results
    
    print()
    print("Comparison Results (y_log_loss_mean - lower is better):")
    print("-" * 70)
    for method_name, results in results_comparison.items():
        log_loss = results.get('y_log_loss_mean', 'N/A')
        if isinstance(log_loss, float):
            print(f"  {method_name:30s}: {log_loss:.6f}")
        else:
            print(f"  {method_name:30s}: {log_loss}")
    print()
    
    return results_comparison

def demo_multiple_patterns():
    """
    Test the same imputation method across different missingness patterns.
    """
    print("=" * 70)
    print("DEMO: Testing Across Missingness Patterns")
    print("=" * 70)
    print()
    
    # Create simulation study
    study = SimulationStudy(
        n=100,
        p=10,
        num_runs=1000,
        continuous_pct=0.4,
        integer_pct=0.4,
        sparsity=0.3,
        include_interactions=True,
        include_nonlinear=True,
        include_splines=False,
        seed=42
    )
    
    # Test multiple patterns
    patterns = [
        MCARPattern(),
        MARPattern(),
        MNARPattern()
    ]
    method = MeanImputation()
    
    print(f"Imputation Method: {method.name}")
    print(f"Testing {len(patterns)} missingness patterns:")
    print()
    
    results_comparison = {}
    
    for pattern in patterns:
        print(f"  Running {pattern.name}...")
        rng = default_rng(42)
        results = study.run_scenario(pattern, method, run_rng=rng)
        results_comparison[pattern.name] = results
    
    print()
    print("Comparison Results (y_log_loss_mean - lower is better):")
    print("-" * 70)
    for pattern_name, results in results_comparison.items():
        log_loss = results.get('y_log_loss_mean', 'N/A')
        if isinstance(log_loss, float):
            print(f"  {pattern_name:30s}: {log_loss:.6f}")
        else:
            print(f"  {pattern_name:30s}: {log_loss}")
    print()
    
    return results_comparison

def main():
    """
    Main demo function that runs all demonstrations.
    """
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "Include Y in MVI - Demo Script" + " " * 23 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print("This demo script showcases the simulation framework with simple examples.")
    print("For full-scale simulations, use 'run_simulation.py' or 'make simulate'.")
    print()
    
    try:
        # Demo 1: Basic simulation
        demo_basic_simulation()
        
        print("\n" + "=" * 70 + "\n")
        
        # Demo 2: Multiple methods
        demo_multiple_methods()
        
        print("\n" + "=" * 70 + "\n")
        
        # Demo 3: Multiple patterns
        demo_multiple_patterns()
        
        print("=" * 70)
        print("Demo complete!")
        print("=" * 70)
        print()
        print("Next steps:")
        print("  - Run full simulation: python run_simulation.py")
        print("  - Use Makefile: make simulate")
        print("  - Check documentation: docs/")
        print()
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        print("\nMake sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
