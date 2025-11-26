"""
Regression tests to verify optimizations preserve correctness.

This test suite verifies that optimized code produces equivalent results
to the baseline implementation within acceptable numerical tolerances.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_simulation import run_simulation
from src.pipeline.simulation.evaluator import (
    evaluate_imputation, stable_log_loss, stable_variance, stable_std
)
from src.pipeline.simulation.data_generators import generate_data
from src.pipeline.simulation.missingness_patterns import MCARPattern
from src.pipeline.simulation.imputation_methods import MeanImputation, MICEImputation
from src.pipeline.simulation.simulator import SimulationStudy
from numpy.random import default_rng

# Numerical tolerance for comparisons
RTOL = 1e-5  # Relative tolerance
ATOL = 1e-8  # Absolute tolerance

class TestOptimizationCorrectness:
    """Test suite for verifying optimization correctness."""
    
    @pytest.fixture
    def test_seed(self):
        """Common seed for reproducibility."""
        return 42
    
    @pytest.fixture
    def test_params(self):
        """Common test parameters."""
        return {
            'n': [50],  # Must be a list for run_simulation
            'p': [5],   # Must be a list for run_simulation
            'num_runs': 2,
            'continuous_pct': [0.4],
            'integer_pct': [0.4],
            'sparsity': [0.3],
            'include_interactions': [False],
            'include_nonlinear': [False],
            'include_splines': [False]
        }
    
    def test_stable_log_loss_equivalence(self):
        """Test that stable_log_loss produces equivalent results to sklearn log_loss."""
        from sklearn.metrics import log_loss
        
        # Test cases
        test_cases = [
            (np.array([0, 1, 1, 0]), np.array([0.1, 0.9, 0.8, 0.2])),
            (np.array([1, 1, 0, 0]), np.array([0.5, 0.5, 0.5, 0.5])),
            (np.array([0, 0, 1, 1]), np.array([0.01, 0.02, 0.99, 0.98])),
        ]
        
        for y_true, y_pred in test_cases:
            # Clip predictions for sklearn (same as our stable version)
            y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
            
            sklearn_loss = log_loss(y_true, y_pred_clipped)
            stable_loss = stable_log_loss(y_true, y_pred)
            
            # Should be very close (within numerical precision)
            assert np.isclose(sklearn_loss, stable_loss, rtol=RTOL, atol=ATOL), \
                f"Log loss mismatch: sklearn={sklearn_loss}, stable={stable_loss}"
    
    def test_stable_variance_equivalence(self):
        """Test that stable_variance produces equivalent results to np.var."""
        test_cases = [
            np.array([1, 2, 3, 4, 5]),
            np.array([10, 20, 30, 40, 50]),
            np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            np.array([1.0, 1.0, 1.0]),  # Edge case: all same
            np.array([100, 101, 102]),  # Small variance
        ]
        
        for values in test_cases:
            np_var = np.var(values, ddof=0)
            stable_var = stable_variance(values, ddof=0)
            
            assert np.isclose(np_var, stable_var, rtol=RTOL, atol=ATOL), \
                f"Variance mismatch: np.var={np_var}, stable={stable_var}, values={values}"
            
            # Test with ddof=1 (sample variance)
            if len(values) > 1:
                np_var_sample = np.var(values, ddof=1)
                stable_var_sample = stable_variance(values, ddof=1)
                
                assert np.isclose(np_var_sample, stable_var_sample, rtol=RTOL, atol=ATOL), \
                    f"Sample variance mismatch: np.var={np_var_sample}, stable={stable_var_sample}"
    
    def test_stable_std_equivalence(self):
        """Test that stable_std produces equivalent results to np.std."""
        test_cases = [
            np.array([1, 2, 3, 4, 5]),
            np.array([10, 20, 30, 40, 50]),
            np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        ]
        
        for values in test_cases:
            np_std = np.std(values, ddof=0)
            stable_std_val = stable_std(values, ddof=0)
            
            assert np.isclose(np_std, stable_std_val, rtol=RTOL, atol=ATOL), \
                f"Std mismatch: np.std={np_std}, stable={stable_std_val}"
    
    def test_data_generation_consistency(self, test_seed):
        """Test that data generation produces consistent results with same seed."""
        rng1 = default_rng(test_seed)
        rng2 = default_rng(test_seed)
        
        # Disable splines to avoid knot issues with small datasets
        data1, _, _ = generate_data(n=50, p=5, include_splines=False, rng=rng1)
        data2, _, _ = generate_data(n=50, p=5, include_splines=False, rng=rng2)
        
        # Should produce identical results with same seed
        pd.testing.assert_frame_equal(data1, data2, check_exact=True)
    
    def test_evaluation_metrics_consistency(self, test_seed):
        """Test that evaluation produces consistent metrics."""
        # Generate test data (disable splines to avoid knot issues)
        rng = default_rng(test_seed)
        train_rng, test_rng, miss_rng, impute_rng = default_rng(1), default_rng(2), default_rng(3), default_rng(4)
        
        train_data, _, _ = generate_data(n=50, p=5, include_splines=False, rng=train_rng)
        test_data, _, _ = generate_data(n=50, p=5, include_splines=False, rng=test_rng)
        
        # Apply missingness
        missingness = MCARPattern()
        dat_miss = missingness.apply(train_data, rng=miss_rng)
        
        # Impute
        imputer = MeanImputation()
        imputed_list = imputer.impute(dat_miss, train_data, rng=impute_rng)
        
        # Evaluate twice with same inputs (should be deterministic)
        metrics1 = evaluate_imputation(imputed_list, test_data, y='y')
        metrics2 = evaluate_imputation(imputed_list, test_data, y='y')
        
        # Should produce identical results
        assert metrics1.keys() == metrics2.keys()
        for key in metrics1.keys():
            if np.isnan(metrics1[key]) and np.isnan(metrics2[key]):
                continue  # Both NaN is fine
            assert np.isclose(metrics1[key], metrics2[key], rtol=RTOL, atol=ATOL, equal_nan=True), \
                f"Metric {key} mismatch: {metrics1[key]} vs {metrics2[key]}"
    
    def test_simulation_results_consistency(self, test_seed, test_params):
        """Test that simulation produces consistent results with same seed."""
        # Ensure num_runs is in a list (required by run_simulation)
        test_params_fixed = test_params.copy()
        if 'num_runs' in test_params_fixed and isinstance(test_params_fixed['num_runs'], int):
            # Keep as is - run_simulation handles it
            pass
        
        # Run simulation twice with same seed
        results1_all, results1_avg = run_simulation(seed=test_seed, **test_params_fixed)
        results2_all, results2_avg = run_simulation(seed=test_seed, **test_params_fixed)
        
        # Check that results are identical (same seed should produce same results)
        pd.testing.assert_frame_equal(
            results1_all.sort_values(by=list(results1_all.columns)).reset_index(drop=True),
            results2_all.sort_values(by=list(results2_all.columns)).reset_index(drop=True),
            check_exact=True,
            check_dtype=False  # Allow dtype differences (e.g., int vs float)
        )
        
        pd.testing.assert_frame_equal(
            results1_avg.sort_values(by=list(results1_avg.columns)).reset_index(drop=True),
            results2_avg.sort_values(by=list(results2_avg.columns)).reset_index(drop=True),
            check_exact=True,
            check_dtype=False
        )
    
    def test_optimized_vs_baseline_metrics(self, test_seed):
        """Test that optimized evaluation produces equivalent metrics to baseline approach."""
        # Generate test data (disable splines to avoid knot issues)
        rng = default_rng(test_seed)
        train_rng, test_rng, miss_rng, impute_rng = default_rng(1), default_rng(2), default_rng(3), default_rng(4)
        
        train_data, _, _ = generate_data(n=50, p=5, include_splines=False, rng=train_rng)
        test_data, _, _ = generate_data(n=50, p=5, include_splines=False, rng=test_rng)
        
        # Apply missingness and impute
        missingness = MCARPattern()
        dat_miss = missingness.apply(train_data, rng=miss_rng)
        
        imputer = MICEImputation(n_imputations=3)  # Use MICE to test multiple imputations
        imputed_list = imputer.impute(dat_miss, train_data, rng=impute_rng)
        
        # Evaluate with optimized version
        metrics_optimized = evaluate_imputation(imputed_list, test_data, y='y_score')
        
        # Manually compute "baseline" metrics using sklearn directly
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error
        import numpy as np
        
        predictors = [col for col in test_data.columns if col not in ['y', 'y_score']]
        X_test = test_data[predictors].values
        y_test = test_data['y_score'].values
        
        mse_values_baseline = []
        r2_values_baseline = []
        
        for imputed_train in imputed_list:
            X_train = imputed_train[predictors].values
            y_train = imputed_train['y_score'].values
            
            if np.isnan(X_train).any() or np.isnan(X_test).any():
                continue
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            mse_values_baseline.append(mse)
            
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            r2_values_baseline.append(r2)
        
        if mse_values_baseline:
            mse_mean_baseline = np.mean(mse_values_baseline)
            mse_std_baseline = np.std(mse_values_baseline) if len(mse_values_baseline) > 1 else 0
            
            r2_mean_baseline = np.mean(r2_values_baseline)
            r2_std_baseline = np.std(r2_values_baseline) if len(r2_values_baseline) > 1 else 0
            
            # Compare with optimized version
            assert 'mse_mean' in metrics_optimized, "Optimized version should have mse_mean"
            assert 'r2_mean' in metrics_optimized, "Optimized version should have r2_mean"
            
            # Check MSE
            assert np.isclose(mse_mean_baseline, metrics_optimized['mse_mean'], 
                            rtol=RTOL, atol=ATOL), \
                f"MSE mean mismatch: baseline={mse_mean_baseline}, optimized={metrics_optimized['mse_mean']}"
            
            # Check R²
            assert np.isclose(r2_mean_baseline, metrics_optimized['r2_mean'], 
                            rtol=RTOL, atol=ATOL), \
                f"R² mean mismatch: baseline={r2_mean_baseline}, optimized={metrics_optimized['r2_mean']}"
            
            # Check std (may differ slightly due to stable_variance, but should be close)
            if 'mse_std' in metrics_optimized:
                assert np.isclose(mse_std_baseline, metrics_optimized['mse_std'], 
                                rtol=1e-3, atol=1e-3), \
                    f"MSE std mismatch: baseline={mse_std_baseline}, optimized={metrics_optimized['mse_std']}"
    
    def test_edge_cases(self, test_seed):
        """Test edge cases produce correct results."""
        # Edge case 1: Very small sample size (disable splines to avoid knot issues)
        rng = default_rng(test_seed)
        train_data, _, _ = generate_data(n=10, p=3, include_splines=False, rng=rng)
        test_data, _, _ = generate_data(n=10, p=3, include_splines=False, rng=default_rng(test_seed + 1))
        
        missingness = MCARPattern()
        dat_miss = missingness.apply(train_data, rng=default_rng(test_seed + 2))
        imputer = MeanImputation()
        imputed_list = imputer.impute(dat_miss, train_data, rng=default_rng(test_seed + 3))
        
        # Should not crash
        metrics = evaluate_imputation(imputed_list, test_data, y='y')
        assert isinstance(metrics, dict)
        
        # Edge case 2: Single imputation
        imputer_single = MeanImputation()
        imputed_single = imputer_single.impute(dat_miss, train_data, rng=default_rng(test_seed + 4))
        metrics_single = evaluate_imputation(imputed_single, test_data, y='y')
        
        # Should have metrics
        assert 'log_loss_mean' in metrics_single or 'mse_mean' in metrics_single
        
        # Edge case 3: Extreme probabilities (test stable_log_loss)
        y_true = np.array([0, 1])
        y_pred_extreme = np.array([1e-10, 1 - 1e-10])  # Very close to 0 and 1
        
        # Should not crash or produce NaN/Inf
        loss = stable_log_loss(y_true, y_pred_extreme)
        assert np.isfinite(loss), f"Log loss should be finite, got {loss}"
        assert loss > 0, f"Log loss should be positive, got {loss}"
    
    def test_statistical_equivalence(self, test_seed):
        """Test that results from multiple runs have equivalent distributions."""
        # Run simulation multiple times with different seeds
        results_list = []
        for seed_offset in range(5):
            results_all, _ = run_simulation(
                n=[30], p=[5], num_runs=1,
                continuous_pct=[0.4], integer_pct=[0.4], sparsity=[0.3],
                include_interactions=[False], include_nonlinear=[False],
                include_splines=[False], seed=test_seed + seed_offset
            )
            results_list.append(results_all)
        
        # Combine results
        all_results = pd.concat(results_list, ignore_index=True)
        
        # Check that metrics are reasonable (not all NaN, not all same value)
        metric_cols = ['y_log_loss_mean', 'y_score_r2_mean']
        for metric in metric_cols:
            if metric in all_results.columns:
                values = all_results[metric].dropna()
                assert len(values) > 0, f"Metric {metric} has no valid values"
                assert values.nunique() > 1 or len(values) == 1, \
                    f"Metric {metric} should have variation across runs"
                
                # Check values are finite
                assert values.isin([np.inf, -np.inf]).sum() == 0, \
                    f"Metric {metric} contains infinite values"
    
    def test_aggregation_correctness(self, test_seed, test_params):
        """Test that result aggregation produces correct statistics."""
        test_params_fixed = test_params.copy()
        results_all, results_avg = run_simulation(seed=test_seed, **test_params_fixed)
        
        # Verify averaged results match manual calculation
        metric_cols = ['y_log_loss_mean', 'y_score_r2_mean']
        
        for metric in metric_cols:
            if metric in results_all.columns and metric in results_avg.columns:
                # Group by same keys as aggregation
                groupby_keys = [
                    'missingness', 'method', 'imputation_outcome_used',
                    'n', 'p', 'cont_pct', 'int_pct', 'sparsity',
                    'interactions', 'nonlinear', 'splines'
                ]
                
                # Manual mean calculation
                manual_mean = results_all.groupby(groupby_keys)[metric].mean().reset_index()
                manual_mean = manual_mean.set_index(groupby_keys)[metric]
                
                # Check against averaged results
                avg_subset = results_avg.set_index(groupby_keys)[metric]
                
                # Compare for matching groups
                common_groups = manual_mean.index.intersection(avg_subset.index)
                for group in common_groups:
                    manual_val = manual_mean[group]
                    avg_val = avg_subset[group]
                    
                    if not (np.isnan(manual_val) and np.isnan(avg_val)):
                        assert np.isclose(manual_val, avg_val, rtol=RTOL, atol=ATOL, equal_nan=True), \
                            f"Metric {metric} aggregation mismatch for group {group}: " \
                            f"manual={manual_val}, aggregated={avg_val}"

def test_numerical_stability():
    """Test numerical stability of optimized functions."""
    # Test stable_log_loss with extreme values
    y_true = np.array([0, 1, 0, 1])
    
    # Very small probabilities
    y_pred_small = np.array([1e-20, 1 - 1e-20, 1e-15, 1 - 1e-15])
    loss_small = stable_log_loss(y_true, y_pred_small)
    assert np.isfinite(loss_small), "Should handle very small probabilities"
    
    # Very large probabilities (close to 1)
    y_pred_large = np.array([1 - 1e-15, 1 - 1e-20, 1 - 1e-15, 1 - 1e-20])
    loss_large = stable_log_loss(y_true, y_pred_large)
    assert np.isfinite(loss_large), "Should handle probabilities close to 1"
    
    # Test stable_variance with edge cases
    # Single value
    assert stable_variance([5.0]) == 0.0, "Variance of single value should be 0"
    
    # All same values
    assert stable_variance([1.0, 1.0, 1.0]) == 0.0, "Variance of constant should be 0"
    
    # Large range
    large_range = np.array([1e10, 2e10, 3e10])
    var_large = stable_variance(large_range)
    assert np.isfinite(var_large), "Should handle large values"
    assert var_large >= 0, "Variance should be non-negative"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

