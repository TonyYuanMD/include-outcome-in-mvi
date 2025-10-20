### Aims
The simulation aims to evaluate how including the outcome variable ($y$ for binary, $y_\text{score}$ for continuous) in the multiple imputation process impacts the effectiveness of missing value imputation (MVI) methods, particularly in recovering missing predictor values and supporting accurate downstream analyses such as regression coefficient estimation. This is assessed using both classical statistical methods and deep learning (DL) methods.

Key questions include:
1. Does incorporating the outcome reduce bias and improve precision in imputed values and downstream estimates under different missingness mechanisms (e.g., MCAR, MAR, MNAR)?
2. How do statistical and DL methods compare in performance when the outcome is included versus excluded?

Hypotheses tested:
1. Including the outcome in MVI enhances effectiveness by preserving relationships between predictors and the outcome, leading to lower bias and better coverage in parameter estimates
2. DL methods outperform statistical ones in complex data scenarios (e.g., with nonlinearity or interactions) when the outcome is included
3. Benefits of including the outcome are more pronounced under MAR or MNAR missingness, where missingness depends on the outcome.

### Data-generating mechanisms
Data is generated via a generalized linear model: For binary $y$, $\text{logits} = X\boldsymbol{\beta}+\text{intercept}$, $y \sim \text{Binomial}(\text{probs}=\frac{1}{1+\exp(-\text{logits})})$; for continuous $y_\text{score} = X\boldsymbol{\beta} + \text{intercept} + \epsilon, (\epsilon \sim \text{N}(0,1))$. Predictors $X_1$ to $X_p$ are mixed: continuous proportion is of `continuous_pct`, following i.i.d $\text{N}(0,1)$, integer proportion is of `integer_pct`, following i.i.d $\text{N}(0,1)$ and rounded, remainder binary following i.i.d $\text{N}(0,1)$ and thresholded with $\text{N}(0,1)>0$, with sparsity proportion of $\beta_i=0$ for some $i\in\{1,\cdots, p\}$. Includes a binary variable`$X_1$_above_threshold` ($\mathbb{1}_{X_1>0}$) to imitate reality. Optional additions: pairwise interactions ($X_i \times X_j$), nonlinear ($\sin$/$\cos$ on continuous), splines (B-spline basis on continuous, degree$=3$, knots at $25/50/75$ percentiles).

Missingness applied to $X_1$ and $X_2$ via logistic probabilities based on `Mmis` coefficients: `mcar` (constant prob $0.2$), `mar` (depends on $X_3$/$X_4$), `mar_type2_y` (depends on $y$), `mar_type2_score` (depends on $y_\text{score}$), MNAR (self-dependent on $X_1$/$X_2$), `mar_threshold` (depends on $X_3$/$X_4$ and `X_1_above_threshold` for $X_2$). Outcomes $y$/$y_\text{score}$ remain complete.

Parameters varying across conditions (defaults in code, customizable):
- `n` (e.g., $500$, $1000$)
- `p` (e.g., $5$, $10$)
- `continuous_pct` (e.g., $0.4$, $0.6$)
- `sparsity` (e.g., $0.3$, $0.5$)
- `include_interactions`/`nonlinear`/`splines` (True/False)
- Missingness patterns (6 types: `mcar`, `mar`, `mar_type2_y`, `mar_type2_score`, `mnar`, `mar_threshold`)

### Estimands/targets
Targets include: true values of missing $X_1$/$X_2$ entries (imputation accuracy); true $\boldsymbol{\beta}$ coefficients and predictive performance for $y$ (binary) and $y_\text{score}$ (continuous) via models fitted on imputed data (downstream utility). Comparisons focus on with/without outcome inclusion in imputation.

### Methods
Methods explicitly vary by outcome inclusion (with $y$, with $y_\text{score}$, without).
Statistical:
1. mean (column means)
2. single (linear regression imputation)
3. MICE (iterative with linear estimator)
4. complete (drop rows with missing $X_1$/$X_2$)

Machine learning:
1. MissForest (iterative RF)
2. MLP (iterative MLP regressor)

Deep learning:
1. AE (autoencoder, PyTorch, with early stopping)
2. GAIN (GAN-based, PyTorch, with early stopping)

For methods supporting it (single, MICE, MissForest, MLP, AE, GAIN), the outcome is optionally concatenated as a feature during imputation. All generate single or multiple imputed datasets; DL uses GPU if available.

### Performance measures
For downstream: After fitting LinearRegression ($y_\text{score}$) or LogisticRegression ($y$) on imputed data, predict on true predictors, compute RMSE/bias ($y_\text{score}$) or log loss/bias ($y$). Aggregated across multiple imputations (pooled mean predictions) and runs, stratified by missingness, method (including with/without outcome), and outcome type. Note: Code handles residual NaNs with mean fallback during evaluation.

### Simulation Design Matrix
Design from code, with factors for feasibility ($2^7\times6\times (2+3\times6)=15360$ combinations in full factorial, but defaults to subset; missingness/methods nested per run).

| Parameter               | Levels/Values                                                                     |
| ----------------------- | --------------------------------------------------------------------------------- |
| `n`                     | 500, 1000                                                                         |
| `p`                     | 5, 10                                                                             |
| `continuous_pct`        | 0.4, 0.6                                                                          |
| `sparsity`              | 0.3, 0.5                                                                          |
| `include_interactions`  | False, True                                                                       |
| `include_nonlinear`     | False, True                                                                       |
| `include_splines`       | False, True                                                                       |
| `missingness_mechanism` | `mcar`, `mar`, `mar_type2_y`, `mar_type2_score`, `mnar`, `mar_threshold`          |
| `include_outcome`       | without, with $y$, with $y_\text{score}$ (exclude outcome-agnostic `method_type`) |
| `method_type`           | Statistical (mean, single, MICE, complete), ML (MissForest, MLP), DL (AE, GAIN)   |
The huge amount of simulations in full factorial design make the experiment computationally costly. To simplify, we can change the design to focus only on the core interest: how different missingness patterns, outcome inclusion, and method types affect imputation performance. This reduce the amount of combinations to $6\times (2+3\times6) =120$.

-----
### Simulation Description for Basic Case (120 Runs)
This simulation evaluates the impact of including the outcome variable ($y$ for binary, $y_\text{score}$ for continuous) in the multiple value imputation (MVI) process on imputation effectiveness and downstream predictive performance across different missingness mechanisms and imputation methods. The focus is on a basic case with fixed data generation parameters, varying only the missingness mechanism, outcome inclusion, and method type, resulting in 144 experimental runs.

#### Data Generation
Data is generated using a generalized linear model:
- For binary $y$, $\text{logits} = X\boldsymbol{\beta}+\text{intercept}$ $y \sim \text{Binomial}(\text{probs}=\frac{1}{1+\exp(-\text{logits})})$
- For continuous $y_\text{score} = X\boldsymbol{\beta} + \text{intercept} + \epsilon, (\epsilon \sim \text{N}(0,1))$.
- Predictors ($X_1$ to $X_5$) include:
  - 40% continuous ($\text{N}(0,1)$),
  - 30% integer (rounded $\text{N}(0,1)$),
  - 30% binary (thresholded $\text{N}(0,1)>0$).
  - $\boldsymbol{\beta}$ coefficients have 30% sparsity (set to $0$).
- Sample size `n` = 1000, number of predictors `p` = 5.
- No interactions, nonlinear terms, or splines are included.

#### Missingness Mechanisms
Missingness is applied to predictors $X_1$ and $X_2$ using logistic probabilities based on predefined mechanisms:
- `MCAR`: Constant probability 0.2.
- `MAR`: Depends on $X_3$ and $X_4$.
- `mar_type2_y`: Depends on $y$.
- `mar_type2_score`: Depends on $y_\text{score}$.
- `MNAR`: Self-dependent on $X_1$ and $X_2$.
- `mar_threshold`: Depends on $X_3$, $X_4$, and a binary indicator $X_1$ > 0 for $X_2$.
- Outcomes $y$ and $y_\text{score}$ remain complete.

#### Imputation Methods
The imputation process employs a comprehensive set of methods as defined in the `impute_datasets` function, with variants based on the inclusion or exclusion of the outcome variables. The full list includes:
- **Outcome-agnostic methods** (no variants):
	- `mean`: Column mean imputation.
	- `complete_data`: Drops rows with missing $X_1$​ or $X_2$​.
- **Variant-supporting methods** (tested with without, with $y$, and with $y_\text{score}$):
    - `single`: Linear regression imputation.
    - `mice`: Iterative MICE with linear estimator.
    - `missforest`: Iterative random forest imputation.
    - `mlp`: Iterative MLP regressor imputation.
    - `ae`: Autoencoder-based imputation.
    - `gain`: GAN-based imputation.

#### Performance Measures
- Downstream utility is assessed by fitting:
  - LinearRegression for $y_\text{score}$ (RMSE, bias).
  - LogisticRegression for $y$ (log loss, bias).

#### Execution
- **Environment**: `Python 3.12+` with `numpy`, `pandas`, `sklearn`, `torch`, `scipy`, `tqdm`, `logging`, `os`.
- **Procedure**:
  1. Set up logging to file and console (INFO level).
  2. Run simulation with fixed parameters mentioned earlier.
  3. For each of the 120 combinations:
     - Generate complete data and save as `complete_data.csv`.
     - Apply missingness pattern to $X_1$ and $X_2$.
     - Impute using each method, saving imputed datasets.
     - Evaluate downstream performance, saving metrics.
  4. Aggregate results across runs, saving `results_all_runs.csv` and `results_averaged.csv` in `results/report/n_1000_p_5_runs_2_cont_0.4_sparse_0.3/`.
- **Command**: Run `run_simulation(n=1000, p=5, num_runs=2, continuous_pct=0.4, sparsity=0.3, include_interactions=False, include_nonlinear=False, include_splines=False, seed=123)`.

This setup allows reproduction of the 120-run basic case, focusing on the interplay of missingness, outcome inclusion, and imputation methods.