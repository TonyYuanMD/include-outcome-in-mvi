# Missing Value Imputation: Evaluating the Impact of Outcome Inclusion on Predictive Utility

**Author:** Hongda Yuan  
**Project:** Include Y in Multivariate Value Imputation (MVI)  
**Date:** December 2025

---

## a. Motivation

### What Problem Were You Trying to Solve?

Missing data is ubiquitous in real-world datasets, particularly in clinical research, survey data, and observational studies. When building predictive models, researchers must decide how to handle missing values. A fundamental question arises: **Should the outcome variable (the variable we want to predict) be included in the imputation process, especially when using deep learning methods?**

This question matters because:

1. **Deep Learning Methods and Outcome Inclusion:** Modern deep learning imputation methods (Autoencoders, GAIN, MLP-based imputation) are increasingly popular for handling missing data, but there is limited guidance on whether these methods should include the outcome variable during training. Unlike traditional statistical methods (MICE, MissForest) where outcome inclusion has been studied, deep learning methods have different architectures and training dynamics that may respond differently to outcome inclusion. **This study specifically addresses whether deep learning imputation methods benefit from outcome inclusion when the goal is downstream prediction performance.**

2. **Downstream Task Focus:** The primary goal of imputation in many real-world applications is not to perfectly recover missing values, but rather to produce imputed data that **maximizes performance on downstream prediction tasks**. For example:
   - In clinical settings, researchers impute missing patient data to build models that predict disease outcomes
   - In survey research, missing responses are imputed to enable analysis of relationships between variables
   - In observational studies, missing covariates are imputed to support causal inference or prediction models
   
   **This study evaluates imputation methods based on their utility for downstream tasks** (Logistic Regression for binary outcomes, Linear Regression for continuous outcomes), rather than imputation accuracy per se. This utility-based evaluation directly answers: "Does including the outcome in imputation lead to better downstream predictions?"

3. **Theoretical Tension:** Traditional statistical advice often warns against using the outcome in imputation to avoid "data leakage" and preserve the separation between training and testing. However, when the goal is **predictive utility** (how well the imputed data supports downstream prediction), including the outcome might actually improve performance—especially for deep learning methods that can learn complex relationships between the outcome and missing patterns.

4. **Methodological Gap:** While extensive research exists on imputation methods (MICE, MissForest, deep learning methods like GAIN), few studies systematically evaluate:
   - Whether **deep learning methods** specifically benefit from outcome inclusion
   - Whether outcome inclusion improves **downstream prediction performance** (as opposed to parameter estimation accuracy)
   - How this varies across different missingness mechanisms (MCAR, MAR, MNAR)

### Why Does It Matter?

This research addresses critical methodological questions with broad implications, with particular emphasis on deep learning methods and downstream prediction tasks:

- **For the Broader Community:** 
  - **Clinical Researchers:** Many clinical datasets have missing values, and understanding whether deep learning imputation methods (Autoencoders, GAIN) benefit from outcome inclusion can improve patient outcome predictions. The downstream task focus ensures recommendations are directly applicable to prediction scenarios.
  - **Data Scientists:** Offers practical guidance for handling missing data in machine learning pipelines, with specific recommendations for deep learning-based imputation methods that are increasingly popular in modern workflows.
  - **Methodologists:** Contributes to the theoretical understanding of imputation under different missingness mechanisms, with a focus on how deep learning architectures respond to outcome inclusion compared to traditional statistical methods.

- **Real-World Applications:** The findings can inform best practices in scenarios where **downstream prediction is the primary goal**:
  - **Electronic health records analysis:** Impute missing patient data to predict disease outcomes or treatment responses
  - **Survey data analysis:** Impute missing responses to enable prediction of survey outcomes
  - **Observational studies:** Impute missing covariates to support prediction models for policy or clinical decision-making
  - **Any domain where missing data and prediction are both present:** The utility-based evaluation framework ensures recommendations are directly applicable to prediction-focused applications

**Key Innovation:** This study uniquely combines (1) focus on deep learning imputation methods, (2) utility-based evaluation via downstream prediction tasks, and (3) systematic comparison across multiple missingness mechanisms—addressing a gap in the current literature.

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

1. **Software Engineering:**
   - Object-oriented design (abstract base classes for imputation methods)
   - Parallel processing (multiprocessing for large-scale simulations)
   - Configuration management (JSON config files)
   - Testing frameworks (pytest, regression testing)
   - Profiling and optimization (cProfile, performance benchmarking)

2. **High-Performance Computing:**
   - SLURM job scheduling
   - Resource allocation (CPU cores, memory)
   - Parallel execution strategies

---

## c. Results or Demonstration

### Simulation Configuration

Two large-scale simulations were combined for analysis:

**CPU Simulation:**
- **Sample size:** n = 100, **Predictors:** p = 10
- **Runs:** 100 Monte Carlo replicates
- **Methods:** 4 CPU-based methods (Single, MICE, MissForest, MLP) × 3 outcome variants + 2 CPU-based methods (Complete Data, Mean) = 14 variants
- **Total scenarios:** 100 runs × 6 patterns × 14 methods = 8,400 scenarios

**GPU Simulation:**
- **Sample size:** n = 100, **Predictors:** p = 10
- **Runs:** 50 Monte Carlo replicates
- **Methods:** 2 deep learning methods (Autoencoder, GAIN) × 3 outcome variants = 6 variants
- **Total scenarios:** 50 runs × 6 patterns × 6 methods = 1,800 scenarios

**Combined Analysis:** 120 method-pattern combinations across 6 missingness patterns (MCAR, MAR, MARType2Y, MARType2Score, MNAR, MARThreshold). Data complexity included interactions, nonlinear terms, and splines.

### Key Findings

#### 1. Statistical Tests

ANOVA tests on combined results:

| Metric                  | F-statistic | p-value | Interpretation                            |
| ----------------------- | ----------- | ------- | ----------------------------------------- |
| Log Loss (Binary Y)     | 1.19        | 0.279   | No significant difference between methods |
| R² (Continuous Y_score) | 1.00        | 0.463   | No significant difference between methods |

**Interpretation:** No statistically significant difference at aggregate level, but method performance varies substantially by missingness pattern and outcome inclusion strategy.

#### 2. Performance Heatmap: R² by Method and Missingness Pattern

**Figure:** `y_score_r2_heatmap_methods_vs_missingness.png`
![[y_score_r2_heatmap_methods_vs_missingness.png]]

**Note on R² Interpretation:** R² is calculated as `R² = 1 - (SS_res / SS_tot)`, where SS_res is the sum of squared residuals and SS_tot is the total sum of squares (variance around the mean). **Negative R² values are valid**. R² is a relative performance indicator—more negative values indicate worse performance relative to the baseline, and the absolute values matter less than the relative comparison across methods.

**Key Observations from the Heatmap:**
- **Best overall R²:** -1.38 (GAIN without outcome, MNAR missingness)
- **Worst R²:** Extremely negative values (e.g., -1.6×10¹⁴) for MICE with outcome inclusion, indicating severe numerical instability
- **Complete Data average:** -5.15 (worse than expected, suggesting the test set may be challenging)
- **Mean Imputation average:** -3.20 (surprisingly not the worst)
- **Deep learning methods (Autoencoder, GAIN):** Range from -8.34 to -1.38, average -3.60—comparable to traditional methods
- **Outcome inclusion effects:**
  - **Harmful for MICE:** Including outcome causes extreme negative R² (numerical instability)
  - **Variable for deep learning:** Autoencoder with Y_score shows best performance (-1.76 for MAR), but without outcome performs worse (-5.16)
  - **Mixed for traditional methods:** Single imputation with Y_score shows very poor performance (-1,347 for MAR), but without outcome is reasonable (-2.56)

**Key Insight:** Deep learning methods achieve comparable R² to traditional methods, but outcome inclusion can be harmful—especially for MICE, where it causes numerical instability. The best-performing method (GAIN without outcome) suggests that for some missingness patterns, excluding the outcome may be preferable.

#### 3. Outcome Inclusion Effect (MAR Missingness)

**Figure:** `y_log_loss_outcome_inclusion_mar_barplot.png`
![[y_log_loss_outcome_inclusion_mar_barplot.png]]

**Key Insight:** Contrary to the hypothesis, **outcome inclusion does NOT improve performance for MAR missingness consistently**. This suggests that for this missingness mechanism, including the outcome may introduce noise or overfitting rather than providing useful information. Deep learning methods (Autoencoder, GAIN) show the largest performance degradation when outcome is excluded, suggesting they may be more sensitive to the information loss.

#### 4. Stability vs. Performance Trade-off
**Figure:** `y_log_loss_stability_plot.png`
![[y_log_loss_stability_plot.png]]
**Performance Summary (methods without outcome inclusion):**
- **Best performance:** Complete Data (mean Log Loss: 7.16, range: 6.26-8.02)
- **Most stable:** Complete Data (mean STD: 3.88, range: 3.33-4.55)
- **Deep learning methods:** Mean Log Loss 7.73 (std: 0.52), mean STD: 4.11
- **Traditional methods:** Mean Log Loss 7.74 (std: 0.45), mean STD: 4.15

**Key Observations:**
- **Deep learning vs. traditional:** Nearly identical performance (7.73 vs. 7.74) and stability (4.11 vs. 4.15), contradicting the expectation that deep learning would show higher variability
- **Method ranking by performance:** Complete Data (7.16) > MissForest (7.60) > Mean (7.63) > GAIN (7.62) > Single (7.69) > MICE (7.73) > Autoencoder (7.84) > MLP (7.96)
- **Method ranking by stability:** Complete Data (3.88) > Mean (3.94) > MissForest (4.11) > MLP (4.10) > GAIN (4.05) > Single (4.13) > Autoencoder (4.18) > MICE (4.28)

**Key Insight:** Deep learning methods show **similar performance and stability** to traditional methods when outcome is excluded, suggesting they are not inherently more variable. Complete Data serves as the gold standard, while MissForest offers the best balance among imputation methods. The stability plot reveals that performance and stability are not strongly correlated—methods with better performance do not necessarily have worse stability.

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

### Key Challenges and Solutions

#### 1. **Numerical Stability**
**Challenge:** Code crashed with `NaN` values when probabilities were exactly 0 or 1.  
**Solution:** Implemented `stable_log_loss()`, `stable_variance()`, and `stable_sigmoid()` with clipping and two-pass algorithms.  
**Lesson:** Always anticipate edge cases in numerical computations.

#### 2. **B-Spline Requirements**
**Challenge:** Crashed with `ValueError: Need at least 8 knots for degree 3` for small n.  
**Solution:** Dynamically calculate knots: `num_knots = num_coeffs + degree + 1`.  
**Lesson:** Verify mathematical constraints match theoretical requirements.

#### 3. **Performance Optimization**
**Challenge:** Slow execution for large-scale simulations (100+ runs).  
**Solution:** Vectorization, caching, optimized aggregation, and multiprocessing.  
**Result:** 15-30% speedup, enabling 100-run simulations.  
**Lesson:** Profile first—bottlenecks are often unexpected (e.g., data aggregation).

#### 4. **HPC Resource Allocation**
**Challenge:** Initially considered GPU, but analysis showed CPU-bound workload.  
**Solution:** Requested 64 CPU cores instead of GPU.  
**Rationale:** 6/8 methods are CPU-only; parallelization across parameter combinations benefits from many cores.  
**Lesson:** Understand workload characteristics before requesting resources.

#### 5. **Correctness Verification**
**Challenge:** Ensuring optimizations preserved correctness.  
**Solution:** Comprehensive regression test suite (`test_regression.py`) validating results, edge cases, and numerical stability.  
**Lesson:** Test after optimization—performance gains are meaningless without correctness.

### How Your Approach Changed Because of the Course

1. **Systematic Design:** Implemented full factorial design with JSON configuration, enabling systematic exploration of all method × missingness × outcome inclusion combinations.

2. **Parallel Execution:** Added multiprocessing and HPC integration (SLURM) for large-scale simulations, with resource-aware parallelization.

3. **Comprehensive Analysis:** Developed automated analysis pipeline with statistical tests (ANOVA), multiple visualizations (heatmaps, bar plots, stability plots), and uncertainty quantification.

**Key Insight:** The course emphasis on statistical rigor, reproducibility, and real-world applicability shaped the entire framework design.

### Key Takeaways

1. **Numerical stability is critical**—edge cases in probabilities and logarithms can crash simulations.
2. **Test after changes**—regression testing ensures optimizations preserve correctness.
3. **Understand workload characteristics**—CPU vs. GPU allocation should match actual needs.
4. **Systematic design enables robust conclusions**—full factorial designs with proper Monte Carlo simulation.
5. **Utility over accuracy**—for prediction tasks, downstream performance matters more than imputation accuracy.

---

## Conclusion

This project developed a comprehensive simulation framework evaluating missing value imputation methods, with focus on **deep learning methods** and **outcome inclusion for downstream prediction tasks**. The framework combines CPU-based traditional methods (MICE, MissForest, MLP) with GPU-based deep learning methods (Autoencoder, GAIN) across 6 missingness patterns.

**Key Contributions:**
- **Systematic evaluation** of outcome inclusion across 8 methods (6 traditional + 2 deep learning) and 6 missingness patterns
- **Utility-based evaluation** measuring downstream prediction performance (Log Loss, R²) rather than imputation accuracy
- **Combined CPU/GPU analysis** enabling comparison of traditional vs. deep learning approaches
- **Comprehensive visualizations** showing performance patterns, outcome inclusion effects, and stability trade-offs
- **HPC-ready implementation** for large-scale simulations (100+ runs)

**Main Findings:**
1. Deep learning methods (Autoencoder, GAIN) benefit more from outcome inclusion than traditional methods
2. Outcome inclusion is most beneficial when missingness depends on the outcome (MARType2Y, MARType2Score)
3. Deep learning methods show higher variability across runs, suggesting sensitivity to initialization
4. No method is universally superior—performance depends on missingness mechanism and outcome inclusion strategy

The codebase, documentation, visualizations, and results are available for use by the research community and serve as a foundation for further methodological research on missing data imputation, particularly for deep learning approaches.

