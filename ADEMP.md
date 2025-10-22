### Aims

The simulation aims to evaluate how including the outcome variable ($\text{y}$ for binary, $\text{y\_score}$ for continuous) in the multiple imputation process impacts the effectiveness of Missing Value Imputation (MVI) methods, specifically by measuring the **predictive utility** of the resulting imputed datasets for downstream regression tasks.

Key questions include:

1. Does incorporating the outcome improve the predictive accuracy ($\text{Log Loss}$ or $R^2$) of the downstream model under different missingness mechanisms ($\text{MCAR}$, $\text{MAR}$, $\text{MNAR}$)?
    
2. How does the **stability** ($\text{STD}$ across runs) of statistical and deep learning (DL) methods compare when the outcome is included versus excluded?
    

Hypotheses tested:

1. Including the outcome in MVI enhances predictive utility, leading to lower $\text{Log Loss}$ for binary $Y$ and higher $R^2$ for continuous $Y_{\text{score}}$.
    
2. Benefits of including the outcome are more pronounced under MAR or MNAR missingness, where missingness depends on the outcome.
    

### Data-Generating Mechanisms (DGM)

Data is generated via a generalized linear model with controllable complexity:

- **Binary** $\mathbf{Y}$**:** $\text{logits} = X\boldsymbol{\beta}+\text{intercept}$, $Y \sim \text{Binomial}(\text{probs}=\frac{1}{1+\exp(-\text{logits})})$.
    
- **Continuous** $\mathbf{Y_{score}}$**:** $Y_{\text{score}} = X\boldsymbol{\beta}+\text{intercept} + \epsilon$.
    
- **Covariates:** Configurable mix of continuous, integer, and binary variables.
    
- **Complexity:** Optional inclusion of interaction terms, non-linear terms ($\text{sin}/\text{cos}$), and spline basis expansions.
    
- **Sparsity:** Coefficients ($\boldsymbol{\beta}$) are generated with a configurable proportion of zero values.
    

### Missingness Mechanisms

Missingness is applied only to $\mathbf{X_1}$ and $\mathbf{X_2}$. The mechanisms tested are:

- $\text{MCAR}$
    
- $\text{MAR}$ (depends on $\text{X3}$, $\text{X4}$)
    
- $\text{MARType2Y}$ (depends on binary $\text{Y}$)
    
- $\text{MARType2Score}$ (depends on continuous $\text{Y}_{\text{score}}$)
    
- $\text{MNAR}$ (depends on the missing value itself)
    
- $\text{MARThreshold}$ (depends on other variables when the missing variable is below a threshold)
    

### Imputation Methods and Outcome Inclusion

Each method is tested in three variants regarding outcome inclusion:

1. $\text{\_without}$**:** Outcome is excluded from the imputation model's predictors.
    
2. $\text{\_with\_y}$**:** Binary outcome ($\text{Y}$) is included as a predictor.
    
3. $\text{\_with\_y\_score}$**:** Continuous outcome ($\text{Y}_{\text{score}}$) is included as a predictor.
    

**Methods:** $\text{complete\_data}$, $\text{mean}$, $\text{single}$, $\text{mice}$, $\text{missforest}$, $\text{mlp}$, $\text{ae}$ (Autoencoder), $\text{gain}$ (GAN).

### Performance Measures (Utility-Based Evaluation)

Downstream utility is assessed by training the final prediction model on the **imputed training data** and evaluating its performance on an **independent, complete test set**.

- **For Binary** $\mathbf{Y}$**:** Utility is assessed by fitting **Logistic Regression** and measuring **Mean Log Loss** (Performance) and the $\text{STD}$ of $\text{Log Loss}$ across runs (Stability).
    
- **For Continuous** $\mathbf{Y_{score}}$**:** Utility is assessed by fitting **Linear Regression** and measuring **Mean** $R^2$ (Performance) and the $\text{STD}$ of $R^2$ across runs (Stability).
    

### Execution and Reproducibility

- **Execution Workflow:** The pipeline is executed using the `Makefile` targets.
    
- **Data Splitting:** A new, complete test set is generated for evaluation in every scenario run.
    
- **Aggregation:** Results are averaged across $\text{n\_imputations}$ (Imputation Uncertainty) and $\text{num\_runs}$ (Simulation Uncertainty).
    
- **Outputs:** `results_all_runs.csv` and `results_averaged.csv` are saved in the parameter-specific report directory.