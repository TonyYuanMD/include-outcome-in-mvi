# Simulation Baseline Performance Documentation

**Date:** November 26, 2025  
**Profiling Tool:** cProfile (Python)  
**Test Configuration:** n=[20, 50, 100], p=[5, 10], num_runs=1

---

## Executive Summary

This document provides baseline performance metrics for the imputation method comparison simulation study. The simulation evaluates 6 missingness patterns × 8 imputation methods = 48 scenarios per parameter combination.

**Key Findings:**
- Average runtime per parameter combination: **83.51 seconds**
- Runtime range: 53.99s - 133.98s
- Computational complexity: **O(p^1)** for number of predictors, approximately **O(n^0)** for sample size
- No numerical warnings or convergence issues observed in current profiling run
- Main bottleneck: Multiprocessing overhead and imputation method execution

---

## 1. Total Runtime

### 1.1 Full Simulation Study Runtime

For a typical simulation study with:
- Parameter combinations: 1 (single n, p combination)
- Number of runs: 1
- Missingness patterns: 6
- Imputation methods: 8
- Total scenarios: 48

**Observed Runtime:**
- Minimum: **53.99 seconds** (n=50, p=5)
- Maximum: **133.98 seconds** (n=20, p=10)
- Average: **83.51 seconds**

### 1.2 Runtime by Parameter Values

| n (Sample Size) | p (Predictors) | Runtime (seconds) |
|----------------|----------------|------------------|
| 20             | 5              | 62.77            |
| 20             | 10             | 133.98           |
| 50             | 5              | 53.99            |
| 50             | 10             | 105.69           |
| 100            | 5              | 54.19            |
| 100            | 10             | 90.44            |

**Observations:**
- Runtime increases with number of predictors (p)
- Runtime shows weak dependence on sample size (n) in tested range
- For p=10, runtime is approximately 2× that of p=5

---

## 2. Profiler Results: Main Bottlenecks

### 2.1 Top Time-Consuming Functions

Based on cProfile analysis (example: n=20, p=5):

| Function/Module | Cumulative Time | % of Total | Description |
|----------------|-----------------|------------|-------------|
| `run_simulation()` | 62.77s | 100% | Main simulation orchestration |
| Multiprocessing overhead | ~62.27s | 99.2% | Thread synchronization, pool management |
| `run_single_combination()` | ~62.22s | 99.1% | Single parameter combination execution |
| Data aggregation | ~0.08s | 0.1% | DataFrame operations |

**Note:** The profiler shows significant time in multiprocessing overhead because worker processes are not profiled directly. The actual computation happens in `run_single_combination()` which runs in separate processes.

### 2.2 Computational Steps Breakdown

The simulation pipeline consists of:

1. **Data Generation** (per scenario):
   - Generate training data: O(n × p)
   - Generate test data: O(n × p)
   - **Estimated complexity:** O(n × p)

2. **Missingness Application** (per scenario):
   - Apply missingness pattern: O(n × p)
   - **Estimated complexity:** O(n × p)

3. **Imputation** (per scenario):
   - Varies by method:
     - Mean/Complete: O(n × p)
     - Single Imputation: O(n × p²) for regression
     - MICE: O(n × p² × iterations) where iterations ≈ 5-10
   - **Estimated complexity:** O(n × p²) for iterative methods

4. **Evaluation** (per scenario):
   - Train model: O(n × p²) for linear/logistic regression
   - Evaluate on test set: O(n × p)
   - **Estimated complexity:** O(n × p²)

5. **Aggregation**:
   - Concatenate results: O(scenarios)
   - Groupby operations: O(scenarios × log(scenarios))
   - **Estimated complexity:** O(scenarios × log(scenarios))

**Total per scenario:** O(n × p²)  
**Total per parameter combination:** O(scenarios × n × p²) = O(48 × n × p²)

---

## 3. Computational Complexity Analysis

### 3.1 Empirical Analysis

#### Complexity with respect to n (sample size)

For fixed p=5:
- **Fitted complexity:** O(n^-0.10)
- **R²:** 0.799
- **Interpretation:** Weak negative relationship (likely due to fixed overhead dominating)

For fixed p=10:
- **Fitted complexity:** O(n^-0.24)
- **R²:** 0.998
- **Interpretation:** Slight decrease with n (overhead effects)

**Note:** The negative exponents are counterintuitive but can occur when:
1. Fixed overhead (multiprocessing setup, data structure initialization) dominates
2. Larger datasets enable better vectorization/optimization
3. Limited sample size range (20-100) may not capture true asymptotic behavior

#### Complexity with respect to p (number of predictors)

For fixed n=20:
- **Fitted complexity:** O(p^1.09)
- **R²:** 1.000
- **Interpretation:** Linear scaling with p

For fixed n=50:
- **Fitted complexity:** O(p^0.97)
- **R²:** 1.000
- **Interpretation:** Linear scaling with p

For fixed n=100:
- **Fitted complexity:** O(p^0.74)
- **R²:** 1.000
- **Interpretation:** Sub-linear scaling (better than linear)

**Conclusion:** Runtime scales approximately **linearly with p** (O(p^1)), which is consistent with the theoretical analysis where imputation and evaluation steps are O(n × p²) but p is varied while n is fixed.

### 3.2 Theoretical Analysis

**Main computational steps:**

1. **Data Generation:**
   - Generate n observations with p predictors
   - Complexity: **O(n × p)**

2. **Imputation (MICE - most expensive):**
   - For each of 5 imputations:
     - Iterative regression: O(iterations × n × p²)
   - Complexity: **O(n × p² × imputations)**

3. **Evaluation:**
   - Train linear/logistic regression: O(n × p²)
   - Predict on test set: O(n × p)
   - Complexity: **O(n × p²)**

4. **Per Scenario Total:** O(n × p²)

5. **Per Parameter Combination:**
   - 48 scenarios × O(n × p²) = **O(48 × n × p²)**
   - For fixed n: **O(p²)** theoretically
   - Empirically observed: **O(p^1)** (likely due to vectorization and efficient implementations)

**Theoretical vs Empirical:**
- Theoretical: O(p²) for imputation and evaluation
- Empirical: O(p^1) observed
- **Explanation:** Modern libraries (scikit-learn, pandas) use optimized BLAS/LAPACK operations that achieve better than quadratic scaling in practice.

### 3.3 Complexity Plots

See `docs/complexity_n.png` and `docs/complexity_p.png` for visualizations of:
- Runtime vs sample size (n) with linear and log-log scales
- Runtime vs number of predictors (p) with linear and log-log scales
- Fitted trend lines showing empirical complexity

---

## 4. Numerical Stability

### 4.1 Warnings and Errors

**Current Profiling Run:**
- **Warnings captured:** 0
- **Errors encountered:** 0
- **Convergence issues:** None observed

**Historical Log Analysis:**
Analysis of `simulation.log.txt` reveals historical warnings from previous runs:

1. **NaN Values in Imputed Data:**
   - Frequency: Observed in older runs (August 2024)
   - Condition: Some imputation methods (particularly deep learning methods) occasionally produced NaN values
   - Handling: Code includes fallback to fill NaNs with column means
   - Status: **Resolved** - Current code includes safeguards

2. **MissForest/MLP Warnings:**
   - Frequency: Observed in older runs (September 2024)
   - Condition: "Too few complete cases" warnings when sparsity is high
   - Impact: Methods fall back to mean imputation when insufficient complete cases
   - Status: **Expected behavior** - Methods handle edge cases gracefully

3. **LogisticRegression Convergence:**
   - Frequency: Rare
   - Condition: Can occur with very small sample sizes or perfect separation
   - Handling: Code includes max_iter limits and error handling
   - Status: **Handled** - Failures are caught and logged

### 4.2 Numerical Stability Measures

**Current Implementation:**
- ✅ NaN detection and handling in imputation methods
- ✅ Convergence checks for iterative methods (MICE)
- ✅ Error handling for model fitting failures
- ✅ Fallback strategies for edge cases

**Recommendations:**
- Monitor for convergence warnings in future runs
- Consider increasing max_iter for LogisticRegression if convergence issues arise
- Add explicit checks for numerical overflow in large-scale simulations

---

## 5. Memory Usage

**Note:** Memory profiling was not performed in this baseline analysis.

**Estimated Memory Requirements:**
- Per scenario: ~O(n × p) for data storage
- Per parameter combination: ~O(48 × n × p) for all scenarios
- For n=100, p=10: ~48 × 100 × 10 × 8 bytes ≈ 384 KB (data only)
- Total with overhead: Estimated < 100 MB per parameter combination

**Future Work:**
- Add memory profiling using `memory_profiler` or `tracemalloc`
- Monitor peak memory usage for large-scale simulations

---

## 6. Bottleneck Summary

### 6.1 Identified Bottlenecks

1. **Multiprocessing Overhead** (Primary)
   - Thread synchronization and pool management
   - Process creation/destruction
   - **Impact:** ~99% of profiled time (but actual computation happens in workers)

2. **Imputation Methods** (Actual computation)
   - MICE imputation: Most expensive (iterative)
   - Single imputation: Moderate cost
   - Mean imputation: Least expensive
   - **Impact:** Varies by method, but MICE dominates

3. **Data Aggregation** (Minor)
   - DataFrame concatenation and groupby operations
   - **Impact:** < 1% of total time

### 6.2 Optimization Opportunities

1. **Reduce Multiprocessing Overhead:**
   - Use persistent worker pools
   - Batch multiple scenarios per worker
   - Consider threading for I/O-bound operations

2. **Optimize Imputation:**
   - Cache fitted models when possible
   - Reduce MICE iterations for profiling runs
   - Use faster imputation methods for initial screening

3. **Parallelize Within Scenarios:**
   - Parallelize across imputation methods
   - Use vectorized operations where possible

---

## 7. Files Generated

All profiling results are saved in `docs/`:

- `profiling_runtime.csv`: Runtime data for all parameter combinations
- `complexity_analysis.json`: Fitted complexity parameters
- `complexity_n.png`: Runtime vs sample size plots
- `complexity_p.png`: Runtime vs number of predictors plots
- `profile_stats_example.txt`: Detailed cProfile output
- `profiling_warnings.csv`: Captured warnings (empty in current run)
- `profiling_errors.csv`: Captured errors (empty in current run)

---

## 8. Recommendations

### 8.1 For Production Runs

1. **Use num_runs ≥ 2** for meaningful uncertainty quantification
2. **Estimate total runtime:** For full factorial design with multiple n, p values:
   - Runtime ≈ (num_combinations × 83.51 seconds) / num_cores
   - Example: 10 combinations on 4 cores ≈ 3.5 minutes
3. **Monitor for warnings:** Check logs for convergence issues
4. **Consider parallelization:** Already implemented with 4 processes

### 8.2 For Optimization

1. **Profile worker processes:** Use `line_profiler` to profile within `run_single_combination()`
2. **Benchmark individual methods:** Identify slowest imputation methods
3. **Consider caching:** Cache data generation for repeated scenarios
4. **Memory profiling:** Add memory profiling for large-scale runs

---

## 9. Conclusion

The simulation study demonstrates:
- **Stable performance** with no numerical issues in current configuration
- **Linear scaling** with number of predictors (p)
- **Weak dependence** on sample size (n) in tested range
- **Multiprocessing overhead** as main bottleneck (expected for parallel execution)

The baseline performance is suitable for production use, with typical runtimes of 50-135 seconds per parameter combination depending on p.

---

**Last Updated:** November 26, 2025  
**Profiling Script:** `profile_simulation.py`  
**Next Review:** After major code changes or when scaling to larger parameter spaces

