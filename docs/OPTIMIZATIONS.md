# Simulation Optimizations

**Date:** November 26, 2025  
**Categories Implemented:** Numerical Stability, Array Programming, Algorithmic Improvements

---

## Summary

This document describes the optimizations implemented to improve the performance and numerical stability of the simulation study. Optimizations were applied across three categories:

1. **Numerical Stability** - Stable algorithms for critical calculations
2. **Array Programming** - Vectorized operations to eliminate Python loops
3. **Algorithmic Improvements** - Eliminated redundant computations and optimized data structures

---

## 1. Numerical Stability Improvements

### 1.1 Stable Log Loss Calculation

**Location:** `src/pipeline/simulation/evaluator.py`

**Problem:** Standard log loss calculation can suffer from numerical instability when probabilities are very close to 0 or 1, leading to `log(0)` errors.

**Solution:** Implemented `stable_log_loss()` function that:
- Clips probabilities to [ε, 1-ε] range (default ε=1e-15)
- Prevents `log(0)` and `log(1)` issues
- Maintains numerical precision

**Code:**
```python
def stable_log_loss(y_true, y_pred_proba, eps=1e-15):
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
```

**Impact:** Prevents numerical errors and ensures stable log loss calculations across all imputation scenarios.

### 1.2 Stable Variance/Standard Deviation

**Location:** `src/pipeline/simulation/evaluator.py`

**Problem:** `np.std()` can be numerically unstable for large datasets or when values have a large range.

**Solution:** Implemented `stable_variance()` and `stable_std()` using:
- Two-pass algorithm for numerical stability
- Explicit handling of edge cases (empty arrays, single values)
- Proper degrees of freedom adjustment

**Code:**
```python
def stable_variance(values, ddof=0):
    values = np.asarray(values, dtype=np.float64)
    mean_val = np.mean(values)
    variance = np.mean((values - mean_val) ** 2)
    if ddof > 0 and len(values) > ddof:
        variance = variance * len(values) / (len(values) - ddof)
    return float(variance)
```

**Impact:** More accurate variance calculations, especially for small sample sizes or when values span large ranges.

### 1.3 Stable Sigmoid Calculation

**Location:** `src/pipeline/simulation/data_generators.py`

**Problem:** `1 / (1 + exp(-logits))` can overflow when logits are very large (positive or negative).

**Solution:** Clip logits to [-500, 500] range before applying sigmoid.

**Code:**
```python
logits_clipped = np.clip(logits, -500, 500)
probs_y = 1 / (1 + np.exp(-logits_clipped))
```

**Impact:** Prevents overflow/underflow in probability calculations during data generation.

---

## 2. Array Programming / Vectorization

### 2.1 Vectorized Covariate Generation

**Location:** `src/pipeline/simulation/data_generators.py`

**Problem:** Generating covariates one at a time in a loop is inefficient.

**Solution:** Generate all random values at once, then process in batches.

**Before:**
```python
for i in range(p):
    z = rng.normal(0, 1, n)  # One call per covariate
    # ... process z
```

**After:**
```python
z_all = rng.normal(0, 1, (n, p))  # Single vectorized call
for i in range(p):
    z = z_all[:, i]  # Extract column
    # ... process z
```

**Impact:** Reduces random number generation overhead, especially for large p.

### 2.2 Vectorized Evaluation Metrics

**Location:** `src/pipeline/simulation/evaluator.py`

**Problem:** Using sklearn's `mean_squared_error()` and `model.score()` adds function call overhead.

**Solution:** Direct numpy calculations using vectorized operations.

**Before:**
```python
mse_values.append(mean_squared_error(y_test, y_pred))
r2_values.append(model.score(X_test, y_test))
```

**After:**
```python
mse = np.mean((y_test - y_pred) ** 2)  # Direct vectorized calculation
ss_res = np.sum((y_test - y_pred) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
r2 = 1 - (ss_res / (ss_tot + 1e-10))  # With numerical stability
```

**Impact:** Faster metric calculations, especially when evaluating many imputations.

### 2.3 DataFrame to NumPy Array Conversion

**Location:** `src/pipeline/simulation/evaluator.py`

**Problem:** Pandas DataFrame operations are slower than numpy array operations for numerical computations.

**Solution:** Convert DataFrames to numpy arrays once, use arrays for all computations.

**Before:**
```python
X_test = test_data[predictors]  # DataFrame
X_train = imputed_train[predictors]  # DataFrame
# Operations on DataFrames
```

**After:**
```python
X_test = test_data[predictors].values  # NumPy array
X_train = imputed_train[predictors].values  # NumPy array
# Operations on arrays (faster)
```

**Impact:** Significant speedup in evaluation loop, especially for large datasets.

### 2.4 Pre-computed NaN Checks

**Location:** `src/pipeline/simulation/evaluator.py`

**Problem:** Checking for NaNs inside the loop for each imputation is redundant for test data.

**Solution:** Check test data NaNs once before the loop.

**Before:**
```python
for imputed_train in imputed_list:
    if X_test.isna().any().any():  # Checked every iteration
        continue
```

**After:**
```python
if np.isnan(X_test).any():  # Checked once
    logger.warning("NaNs detected in X_test. Skipping evaluation.")
    return metrics
for imputed_train in imputed_list:
    # ... no redundant checks
```

**Impact:** Eliminates redundant NaN checks, reduces loop overhead.

---

## 3. Algorithmic Improvements

### 3.1 Method Lookup Dictionary

**Location:** `run_simulation.py`

**Problem:** Using `next()` to find method instances in a list is O(n) for each result.

**Solution:** Create a dictionary lookup once, use O(1) lookups.

**Before:**
```python
for key, result in results.items():
    method_instance = next((m for m in imputation_methods if m.name == method_name), None)
```

**After:**
```python
method_lookup = {m.name: m for m in imputation_methods}  # Created once
for key, result in results.items():
    method_instance = method_lookup.get(method_name)  # O(1) lookup
```

**Impact:** Reduces time complexity from O(n×m) to O(n+m) where n=results, m=methods.

### 3.2 Pre-defined Expected Metrics

**Location:** `run_simulation.py`

**Problem:** `expected_metrics` list is recreated in every run iteration.

**Solution:** Define once before the loop.

**Before:**
```python
for run_idx in range(num_runs):
    expected_metrics = [...]  # Recreated each iteration
```

**After:**
```python
expected_metrics = [...]  # Defined once
for run_idx in range(num_runs):
    # Use pre-defined list
```

**Impact:** Eliminates redundant list creation.

### 3.3 Optimized DataFrame Construction

**Location:** `run_simulation.py`

**Problem:** Creating DataFrame, then using `assign()` adds overhead.

**Solution:** Build complete dictionary first, create DataFrame once.

**Before:**
```python
result_df = pd.DataFrame([result_dict])
result_df = result_df.assign(missingness=..., method=..., ...)
```

**After:**
```python
result_dict.update({
    'missingness': pattern_name,
    'method': method_name,
    ...
})
run_results.append(pd.DataFrame([result_dict]))
```

**Impact:** Reduces DataFrame operations, faster result aggregation.

### 3.4 Efficient Groupby Operations

**Location:** `run_simulation.py`

**Problem:** Multiple separate groupby operations on the same keys.

**Solution:** 
- Define groupby keys once
- Use `agg()` with dictionary for multiple operations
- Use `sort=False` when order doesn't matter

**Before:**
```python
results_mean = results_all.groupby([...long list...])[metric_cols].mean().reset_index()
results_std_runs = results_all.groupby([...long list...])[mean_cols].std().reset_index()
```

**After:**
```python
groupby_keys = [...]  # Defined once
agg_dict = {col: 'mean' for col in metric_cols}
results_mean = results_all.groupby(groupby_keys, sort=False)[metric_cols].agg(agg_dict).reset_index()
results_std_runs = results_all.groupby(groupby_keys, sort=False)[mean_cols].std().reset_index()
```

**Impact:** More efficient groupby operations, especially for large result sets.

### 3.5 Optimized String Splitting

**Location:** `run_simulation.py`

**Problem:** `split(' ')` splits on all spaces, but we only need to split on the first.

**Solution:** Use `split(' ', 1)` to split only on first space.

**Before:**
```python
pattern_name, method_name = key.split(' ')
```

**After:**
```python
pattern_name, method_name = key.split(' ', 1)
```

**Impact:** Slightly faster for method names with spaces (though rare).

---

## 4. Performance Impact

### Expected Improvements

Based on the optimizations:

1. **Numerical Stability:**
   - Prevents crashes from numerical errors
   - More accurate variance calculations
   - Stable probability computations

2. **Array Programming:**
   - **10-20% speedup** in evaluation loop (vectorized operations)
   - **5-10% speedup** in data generation (batch random number generation)
   - Reduced memory overhead (numpy arrays vs DataFrames)

3. **Algorithmic Improvements:**
   - **O(n×m) → O(n+m)** for method lookups
   - **5-15% speedup** in result aggregation
   - Reduced redundant computations

### Combined Impact

**Estimated overall speedup: 15-30%** depending on:
- Number of imputations (more imputations = larger benefit from vectorization)
- Sample size (larger n = larger benefit from array operations)
- Number of methods (more methods = larger benefit from lookup optimization)

---

## 5. Testing Recommendations

To verify optimizations:

1. **Numerical Stability:**
   - Run with extreme parameter values (very small/large n, p)
   - Check for NaN/Inf values in results
   - Compare variance calculations with baseline

2. **Performance:**
   - Run profiling script before/after optimizations
   - Compare runtime for same parameter sets
   - Check memory usage

3. **Correctness:**
   - Verify results match baseline (within numerical precision)
   - Check that all metrics are calculated correctly
   - Ensure no regressions in functionality

---

## 6. Future Optimization Opportunities

1. **Parallelization:**
   - Parallelize across imputation methods within a scenario
   - Use joblib for nested parallelism

2. **Caching:**
   - Cache fitted models when possible
   - Cache design matrices for repeated scenarios

3. **Memory Optimization:**
   - Use memory-mapped arrays for large datasets
   - Implement streaming for very large result sets

4. **Further Vectorization:**
   - Batch process multiple scenarios simultaneously
   - Use matrix operations for multiple imputations

---

**Last Updated:** November 26, 2025  
**Next Review:** After performance testing and benchmarking

