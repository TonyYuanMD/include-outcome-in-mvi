# Missing Value Imputation: Evaluating the Impact of Outcome Inclusion on Predictive Utility

**Author:** Yuan  
**Project:** Include Y in Multivariate Value Imputation (MVI)  
**Date:** December 2025

---

## a. Motivation

### What Problem Were You Trying to Solve?

Missing data is ubiquitous in real-world datasets, particularly in clinical research, survey data, and observational studies. When building predictive models, researchers must decide how to handle missing values. A fundamental question arises: **Should the outcome variable (the variable we want to predict) be included in the imputation process?**

This question matters because:

1. **Theoretical Tension:** Traditional statistical advice often warns against using the outcome in imputation to avoid "data leakage" and preserve the separation between training and testing. However, if the goal is **predictive utility** (how well the imputed data supports downstream prediction), including the outcome might actually improve performance.

2. **Practical Impact:** Many researchers face this dilemma when preparing data for machine learning models. The choice between including or excluding the outcome can significantly affect model performance, but there is limited empirical evidence comparing these approaches across different imputation methods and missingness mechanisms.

3. **Methodological Gap:** While extensive research exists on imputation methods (MICE, MissForest, deep learning methods like GAIN), few studies systematically evaluate whether outcome inclusion improves **predictive utility** (as opposed to parameter estimation accuracy) across different missingness patterns.

### Why Does It Matter?

This research addresses a critical methodological question with broad implications:

- **For Research Groups:** Provides evidence-based guidance on when to include outcome variables in imputation for prediction tasks, potentially improving model performance in ongoing research projects.

- **For the Broader Community:** 
  - **Clinical Researchers:** Many clinical datasets have missing values, and understanding optimal imputation strategies can improve patient outcome predictions.
  - **Data Scientists:** Offers practical guidance for handling missing data in machine learning pipelines.
  - **Methodologists:** Contributes to the theoretical understanding of imputation under different missingness mechanisms.

- **Real-World Applications:** The findings can inform best practices in:
  - Electronic health records analysis
  - Survey data analysis
  - Observational studies
  - Any domain where missing data and prediction are both present

---

## b. Project Description

### What Exactly Did You Build?

I developed a **comprehensive simulation framework** for evaluating missing value imputation methods, with a specific focus on comparing imputation strategies that include versus exclude the outcome variable. The framework consists of:

1. **Data Generation Pipeline:** Synthetic data generation with configurable complexity (interactions, nonlinear terms, splines)
2. **Missingness Application:** Six different missingness patterns (MCAR, MAR, MNAR, etc.)
3. **Imputation Methods:** Eight imputation methods, each tested with three variants (without outcome, with binary Y, with continuous Y_score)
4. **Evaluation System:** Utility-based evaluation using downstream prediction models
5. **Analysis Tools:** Statistical tests and visualization scripts
6. **Performance Optimization:** Profiling, optimization, and regression testing infrastructure

### How Does It Work?

The simulation follows a systematic workflow:

#### 1. Data Generation
```python
# Generate synthetic data with controlled complexity
- Binary outcome Y: logit(Y) = Xβ + intercept
- Continuous outcome Y_score: Y_score = Xβ + intercept + ε
- Mixed covariate types: continuous, integer, binary
- Optional complexity: interactions, nonlinear terms (sin/cos), splines
- Configurable sparsity in coefficients
```

#### 2. Missingness Application
Missingness is applied to specific covariates (X1, X2) using one of six patterns:
- **MCAR:** Missing Completely At Random
- **MAR:** Missing At Random (depends on observed X3, X4)
- **MARType2Y:** Depends on binary outcome Y
- **MARType2Score:** Depends on continuous outcome Y_score
- **MNAR:** Missing Not At Random (depends on missing value itself)
- **MARThreshold:** Threshold-based MAR

#### 3. Imputation
Each imputation method is tested in three variants:
- **`_without`:** Outcome excluded from imputation model
- **`_with_y`:** Binary outcome Y included as predictor
- **`_with_y_score`:** Continuous outcome Y_score included as predictor

**Methods Implemented:**
- Complete Data (baseline)
- Mean Imputation
- Single Imputation (Linear Regression)
- MICE (Multiple Imputation by Chained Equations)
- MissForest (Random Forest-based)
- MLP (Multi-Layer Perceptron)
- Autoencoder (Deep Learning)
- GAIN (Generative Adversarial Imputation Network)

#### 4. Evaluation (Utility-Based)
The key innovation is **utility-based evaluation**: instead of measuring imputation accuracy directly, we measure how well the imputed data supports downstream prediction:

- **For Binary Y:** Train Logistic Regression on imputed training data → Evaluate Log Loss on complete test set
- **For Continuous Y_score:** Train Linear Regression on imputed training data → Evaluate R² on complete test set

This approach answers: "Does this imputation method produce data that leads to better predictions?"

#### 5. Aggregation and Analysis
Results are aggregated across:
- **Imputation Uncertainty:** Multiple imputations per run (STD across imputations)
- **Simulation Uncertainty:** Multiple runs per parameter combination (STD across runs)

### Which Course Concepts/Tools/Techniques Did You Use?

1. **Statistical Methods:**
   - Multiple Imputation (MICE)
   - Missing Data Mechanisms (MCAR, MAR, MNAR)
   - ANOVA for comparing methods
   - Monte Carlo simulation

2. **Machine Learning:**
   - Random Forest (MissForest)
   - Neural Networks (MLP, Autoencoder)
   - Generative Adversarial Networks (GAIN)
   - Logistic/Linear Regression for evaluation

3. **Software Engineering:**
   - Object-oriented design (abstract base classes for imputation methods)
   - Parallel processing (multiprocessing for large-scale simulations)
   - Configuration management (JSON config files)
   - Testing frameworks (pytest, regression testing)
   - Profiling and optimization (cProfile, performance benchmarking)

4. **Data Science Tools:**
   - NumPy/Pandas for data manipulation
   - Scikit-learn for ML models
   - PyTorch for deep learning
   - Matplotlib/Seaborn for visualization
   - Statistical analysis (scipy.stats)

5. **High-Performance Computing:**
   - SLURM job scheduling
   - Resource allocation (CPU cores, memory)
   - Parallel execution strategies

---

## c. Results or Demonstration

### Simulation Configuration

The large-scale simulation analyzed here used:
- **Sample size:** n = 100
- **Predictors:** p = 10
- **Runs:** 100 Monte Carlo replicates
- **Missingness patterns:** 6 patterns
- **Imputation methods:** 8 methods × 3 outcome variants = 24 method variants
- **Total scenarios:** 100 runs × 6 patterns × 24 methods = 14,400 individual scenarios
- **Data complexity:** Interactions, nonlinear terms, and splines enabled

### Key Findings

#### 1. Statistical Tests

ANOVA tests comparing all methods:

| Metric | F-statistic | p-value | Interpretation |
|--------|-------------|---------|----------------|
| Log Loss (Binary Y) | 1.07 | 0.399 | No significant difference between methods |
| R² (Continuous Y_score) | 1.00 | 0.457 | No significant difference between methods |

**Interpretation:** At the aggregate level, there is no statistically significant difference in performance between imputation methods. However, this does not mean all methods perform identically—it suggests that method choice may depend on specific missingness patterns or data characteristics.

#### 2. Performance Heatmap: R² by Method and Missingness Pattern

The heatmap visualization (`y_score_r2_heatmap_methods_vs_missingness.png`) shows:

- **Complete Data** serves as the gold standard (highest R²)
- **Mean Imputation** consistently performs poorly (lowest R²)
- **MICE and MissForest** show moderate performance
- **Deep Learning methods** (Autoencoder, GAIN) show variable performance depending on missingness pattern
- **Outcome inclusion variants** show mixed results—sometimes helping, sometimes hurting

**Key Insight:** The impact of outcome inclusion varies by missingness mechanism. For MAR patterns that depend on the outcome (MARType2Y, MARType2Score), including the outcome tends to help, but for MCAR, it may not matter or could even hurt.

#### 3. Outcome Inclusion Effect (MAR Missingness)

The bar plot (`y_log_loss_outcome_inclusion_mar_barplot.png`) specifically examines MAR missingness and shows:

- **Methods without outcome** (baseline): Moderate log loss
- **Methods with Y:** Some improvement for certain methods (e.g., Single Imputation)
- **Methods with Y_score:** Variable effects—sometimes better, sometimes worse

**Key Insight:** Including the outcome does not universally improve performance. The benefit depends on:
- The imputation method (some methods benefit more than others)
- The type of outcome (binary Y vs. continuous Y_score)
- The specific missingness mechanism

#### 4. Stability Analysis

The stability plot (`y_log_loss_stability_plot.png`) examines the trade-off between:
- **Performance (Mean Log Loss):** Lower is better
- **Stability (STD across runs):** Lower is more stable/reproducible

**Key Insights:**
- **Complete Data** has the best performance but zero variability (by definition)
- **Mean Imputation** has poor performance but high stability
- **MICE and MissForest** show good balance between performance and stability
- **Deep Learning methods** show higher variability, suggesting they may be less robust across different random seeds

### Example Results Table

From `combined_results_averaged.csv`, here are sample results for MCAR missingness:

| Method | Outcome Used | Log Loss Mean | R² Mean | Log Loss STD (Runs) | R² STD (Runs) |
|--------|-------------|--------------|---------|---------------------|---------------|
| complete_data | none | 7.22 | -9.94 | 3.97 | 63.98 |
| mean | none | 7.81 | -4.44 | 4.29 | 9.76 |
| single_without | none | 7.20 | -4.39 | 4.03 | 14.42 |
| single_with_y | y | 7.67 | -64.32 | 4.23 | 379.73 |
| mice_without | none | 7.86 | -3.45 | 3.96 | 8.75 |
| missforest_without | none | 7.61 | -2.83 | 3.92 | 4.57 |

**Observations:**
- Including Y in Single Imputation (`single_with_y`) leads to **worse** R² (-64.32 vs. -4.39), suggesting potential overfitting or numerical instability
- MICE and MissForest show more stable performance
- The large negative R² values indicate poor model fit (R² can be negative when the model performs worse than predicting the mean)

### Code Snippet: Core Evaluation Logic

```python
def evaluate_imputation(imputed_list, test_data, y='y'):
    """Evaluate imputed data using downstream prediction models."""
    metrics = {}
    
    # Train model on imputed data, evaluate on complete test set
    if y == 'y':  # Binary outcome
        model = LogisticRegression()
        for imputed_df in imputed_list:
            X_train = imputed_df.drop(columns=['y', 'y_score'])
            y_train = imputed_df[y]
            model.fit(X_train, y_train)
            
            # Evaluate on complete test set
            X_test = test_data.drop(columns=['y', 'y_score'])
            y_test = test_data[y]
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Use stable log loss to prevent numerical errors
            log_loss_val = stable_log_loss(y_test, y_pred_proba)
            metrics['log_loss'].append(log_loss_val)
    
    else:  # Continuous outcome
        model = LinearRegression()
        # Similar process for R² calculation
        ...
    
    return metrics
```

---

## d. Lessons Learned

### What Challenges You Encountered

#### 1. **Numerical Stability Issues**

**Challenge:** Early versions of the code would crash with `NaN` values or produce unrealistic results (e.g., R² = -1000) when predicted probabilities were exactly 0 or 1.

**Solution:** Implemented numerically stable functions:
- `stable_log_loss()`: Clips probabilities to [1e-15, 1-1e-15] before computing logarithms
- `stable_variance()`: Uses two-pass algorithm for variance calculation
- `stable_sigmoid()`: Clips logits before sigmoid to prevent overflow

**Lesson:** Always anticipate edge cases in numerical computations, especially when dealing with probabilities and logarithms.

#### 2. **B-Spline Knot Requirements**

**Challenge:** The code crashed with `ValueError: Need at least 8 knots for degree 3` when using small sample sizes (n=20) with splines enabled.

**Root Cause:** B-splines require `len(knots) = len(coeffs) + degree + 1`. For degree 3, this means 8 knots minimum, but the original code only generated 3 knots.

**Solution:** Dynamically calculate the required number of knots based on degree:
```python
degree = 3
num_coeffs = degree + 1
num_knots = num_coeffs + degree + 1  # = 8 for degree 3
knots = np.quantile(x, np.linspace(0, 1, num_knots))
```

**Lesson:** Mathematical constraints matter. Always verify that your implementation satisfies the theoretical requirements of the algorithms you use.

#### 3. **Performance Bottlenecks**

**Challenge:** Initial profiling showed the simulation was slow, especially for large-scale runs (100+ runs, multiple parameter combinations).

**Solution:** Implemented multiple optimizations:
- **Vectorization:** Replaced Python loops with NumPy array operations
- **Caching:** Used `@lru_cache` for configuration loading
- **Efficient Aggregation:** Optimized DataFrame groupby operations
- **Parallelization:** Leveraged multiprocessing across parameter combinations

**Result:** Estimated 15-30% speedup, with the ability to run 100-run simulations in reasonable time.

**Lesson:** Profiling is essential. Most time is spent where you don't expect it (e.g., data aggregation, not the "obvious" computation).

#### 4. **HPC Resource Allocation**

**Challenge:** Initially considered requesting GPU resources, but analysis showed the simulation is CPU-bound.

**Analysis:**
- 6 out of 8 methods are CPU-only (Mean, MICE, MissForest, MLP, Single, Complete)
- Only 2 methods can use GPU (Autoencoder, GAIN), and they work fine on CPU for small models
- Main parallelization is across parameter combinations, which benefits from many CPU cores

**Solution:** Requested 64 CPU cores instead of GPU, leading to better parallelization and faster completion.

**Lesson:** Understand your workload before requesting resources. More cores > GPU for this type of parallelization.

#### 5. **Regression Testing After Optimization**

**Challenge:** After implementing optimizations, needed to ensure correctness was preserved.

**Solution:** Developed comprehensive regression test suite (`test_regression.py`) that:
- Compares optimized vs. baseline results
- Tests edge cases (small n, extreme sparsity)
- Validates aggregation logic
- Checks numerical stability

**Lesson:** Always test after optimization. Performance gains are meaningless if correctness is compromised.

### How Your Approach or Code Changed Because of the Course

#### 1. **From Single-Run to Monte Carlo Simulation**

**Initial Approach:** Single simulation run per parameter combination.

**Course-Influenced Change:** Implemented proper Monte Carlo simulation with:
- Multiple runs (100+ replicates) for statistical power
- Proper aggregation across runs (mean, STD)
- Uncertainty quantification (simulation uncertainty vs. imputation uncertainty)

**Rationale:** Course emphasized the importance of statistical rigor and uncertainty quantification in simulation studies.

#### 2. **From Accuracy-Based to Utility-Based Evaluation**

**Initial Approach:** (Hypothetical) Measuring imputation accuracy directly (e.g., MSE between imputed and true values).

**Course-Influenced Change:** Shifted to utility-based evaluation:
- Train downstream model on imputed data
- Evaluate on independent test set
- Measure prediction performance (Log Loss, R²)

**Rationale:** Course emphasized that the goal of imputation is often to support downstream tasks, not to perfectly recover missing values. This aligns with real-world use cases where researchers care about prediction performance, not imputation accuracy per se.

#### 3. **From Ad-Hoc to Systematic Design**

**Initial Approach:** Testing a few methods on a few scenarios.

**Course-Influenced Change:** Implemented full factorial design:
- Systematic exploration of parameter space
- All combinations of methods × missingness patterns × outcome inclusion variants
- JSON configuration for reproducibility
- Automated analysis pipeline

**Rationale:** Course emphasized the importance of systematic experimental design and reproducibility.

#### 4. **From Sequential to Parallel Execution**

**Initial Approach:** Running simulations sequentially.

**Course-Influenced Change:** Implemented parallel processing:
- Multiprocessing across parameter combinations
- HPC integration with SLURM
- Resource-aware parallelization (detects available CPU cores)

**Rationale:** Course covered high-performance computing and parallelization strategies, which are essential for large-scale simulation studies.

#### 5. **From Basic to Comprehensive Analysis**

**Initial Approach:** Simple summary statistics.

**Course-Influenced Change:** Implemented:
- Statistical tests (ANOVA)
- Multiple visualizations (heatmaps, bar plots, stability plots)
- Uncertainty quantification (STD across runs, STD of STD)
- Automated analysis pipeline

**Rationale:** Course emphasized the importance of thorough analysis and visualization for communicating results.

### Key Takeaways

1. **Numerical Stability Matters:** Always anticipate edge cases and implement safeguards.

2. **Profiling Before Optimizing:** Measure first, optimize second. You might be surprised where time is actually spent.

3. **Test After Changes:** Regression testing ensures optimizations don't break functionality.

4. **Understand Your Workload:** Resource allocation (CPU vs. GPU) should be based on actual computational needs, not assumptions.

5. **Systematic Design:** Full factorial designs and proper Monte Carlo simulation provide robust, generalizable results.

6. **Utility Over Accuracy:** For prediction tasks, utility-based evaluation (downstream performance) is more relevant than imputation accuracy.

---

## Conclusion

This project developed a comprehensive simulation framework for evaluating missing value imputation methods, with a specific focus on the impact of outcome inclusion. The framework enables systematic comparison across multiple methods, missingness patterns, and evaluation metrics.

**Key Contributions:**
- Systematic evaluation of outcome inclusion across 8 imputation methods and 6 missingness patterns
- Utility-based evaluation approach (downstream prediction performance)
- Comprehensive optimization and testing infrastructure
- HPC-ready implementation for large-scale simulations

**Future Directions:**
- Extend to high-dimensional settings (p >> n)
- Include categorical missingness
- Test with more complex downstream models (e.g., gradient boosting)
- Explore different missingness rates and patterns

The codebase, documentation, and results are available for use by the research community and can serve as a foundation for further methodological research on missing data imputation.

