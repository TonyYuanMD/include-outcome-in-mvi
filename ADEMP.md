### Aims
The simulation aims to evaluate how including the outcome variable (y) in the multiple imputation process impacts the effectiveness of missing value imputation (MVI) methods, particularly in recovering missing predictor values and supporting accurate downstream analyses such as regression coefficient estimation. This is assessed using both classical statistical methods and deep learning (DL) methods. Key questions include: Does incorporating $y$ reduce bias and improve precision in imputed values and downstream estimates under different missingness mechanisms (e.g., MCAR, MAR, MNAR)? How do statistical and DL methods compare in performance when $y$ is included versus excluded? Hypotheses tested: (1) Including $y$ in MVI enhances effectiveness by preserving relationships between predictors and the outcome, leading to lower bias and better coverage in parameter estimates; (2) DL methods outperform statistical ones in complex data scenarios (e.g., with nonlinearity or interactions) when $y$ is included; (3) Benefits of including $y$ are more pronounced under MAR or MNAR missingness, where missingness depends on observed or unobserved data.

### Data-generating mechanisms
The data-generating process involves simulating a linear regression model where the outcome $y$ is a function of $p$ predictors ($X_1$ to $X_p$) plus Gaussian noise: $\boldsymbol{y} =\boldsymbol{X\beta} + \boldsymbol{ε}$, with $\boldsymbol{\beta}$ coefficients incorporating a sparsity level (proportion of zeros). Predictors are mixed: a proportion (`continuous_pct`) are continuous (normal distribution), the remainder integer-valued (e.g., Poisson), and sparsity affects zero-impact predictors. Optional complexities include pairwise interactions ($X_i \times X_j$), nonlinear terms ($\sin(X)$, $\cos(X)$), or spline expansions. Missingness is applied to specific predictors (e.g., $X_1$, $X_2$), using patterns like MCAR (random), MAR (dependent on observed variables), and MNAR (dependent on missing values or $y$). Patterns are defined via masks (`Mmis`) and dependencies (`vars`), ensuring $y$ remains complete for potential inclusion in imputation.

Parameters varying across conditions:
- Sample size ($n$)
- Number of predictors ($p$)
- Proportion of continuous predictors (`continuous_pct`)
- Sparsity level (`sparsity`)
- Inclusion of interactions, nonlinearity, splines (True/False)
- Missingness mechanism (MCAR, MAR, MNAR)
- Inclusion of $y$ in imputation (yes/no)
- Random seed for reproducibility
- Number of runs (`num_runs`) for Monte Carlo averaging

### Estimands/targets
Primary targets are the true missing values in predictors ($X_1$, $X_2$) for imputation accuracy, and downstream quantities like true regression coefficients ($\boldsymbol{\beta}$) and predictive performance for $y$ (e.g., via models fitted on imputed data). Evaluations compare scenarios with and without $y$ in the imputation process, focusing on how inclusion affects recovery of relationships between $X$ and $y$.

### Methods
Statistical methods (from the simulation code and common baselines in the review paper) include:
1. mean/median imputation
2. single imputation by regression
3. multiple imputation by chained equations (MICE, with variants like predictive mean matching)
4. MissForest (random forest-based)
Deep learning methods (drawn from the review paper) include: 
5. autoencoders (AE, denoising AE, variational AE)
6. generative adversarial networks (GANs, e.g., GAIN)
Methods are applied to incomplete datasets, with multiple imputations generated where applicable.

### Performance measures
Metrics for imputation accuracy: bias (imputed vs. true values), variance, mean squared error (MSE), mean absolute error (MAE), root MSE (RMSE). For downstream utility: bias and MSE in estimated $\boldsymbol{\beta}$, prediction MSE for y, coverage probability (proportion of true $\boldsymbol{\beta}$ in confidence intervals), Type I error rate, and power for hypothesis tests on $\boldsymbol{\beta}$. Results are aggregated across runs, with comparisons stratified by whether $y$ is included, missingness type, and method category (stats vs. DL).

### Simulation Design Matrix
The design incorporates factors from the code and review, with a key addition for including y. Assuming 2-3 levels per factor (based on defaults and typical setups), the matrix is a Cartesian product, yielding ~384 combinations (excluding replications). Missingness and methods are nested within each. Subsets may be run for computational feasibility.

| Parameter              | Levels/Values                                                                 |
|------------------------|-------------------------------------------------------------------------------|
| n (observations)      | 500, 1000, 2000                                                               |
| p (predictors)        | 5, 10, 20                                                                     |
| continuous_pct        | 0.4, 0.6                                                                      |
| sparsity              | 0.3, 0.5                                                                      |
| include_interactions  | False, True                                                                   |
| include_nonlinear     | False, True                                                                   |
| include_splines       | False, True                                                                   |
| missingness_mechanism | MCAR, MAR, MNAR                                                               |
| include_y             | No, Yes                                                                       |
| method_type           | Statistical, Deep Learning                                                    |
| Statistical methods   | Mean, Median, kNN, MICE, MissForest, EM, SVD, SoftImpute (within stats)       |
| DL methods            | AE/DAE/VAE, GAN/GAIN, RNN/LSTM/GRU, Transformer, Hybrid (within DL)           |
| num_runs              | 10 (for averaging; adjustable)                                                |
| seed                  | 123 (base; incremented per run)                                               |

Example combinations (subset):
- n=500, p=5, continuous_pct=0.4, sparsity=0.3, interactions=False, nonlinear=False, splines=False, missingness=MCAR, include_y=No, method_type=Statistical
- n=500, p=5, continuous_pct=0.4, sparsity=0.3, interactions=False, nonlinear=False, splines=False, missingness=MCAR, include_y=Yes, method_type=Deep Learning
- ... up to full set.

### Description of the Simulation
To reproduce, use Python 3.12+ with libraries like pandas, tqdm, logging, numpy/scipy (for data gen), scikit-learn/statsmodels (for stats imputation), and PyTorch/TensorFlow (for DL). Custom modules (generate_data, generate_missingness, impute_stats, evaluate_imputations) handle core logic; if unavailable, implement via standard libraries (e.g., generate_data with numpy for linear models; impute_stats with sklearn.impute/pingouin; add impute_dl for DL using torch.nn for AEs/GANs).

1. **Setup**: Configure logging to file/console (INFO level). Define run_simulation with parameters: n=1000, p=5, num_runs=2, continuous_pct=0.4, sparsity=0.3, include_interactions=False, include_nonlinear=False, include_splines=False, include_y=True/False (new flag), seed=123.

2. **Validation**: Ensure integer_pct = 1 - continuous_pct - sparsity >= 0.

3. **Output**: Create directory 'results/report/n_{n}_p_{p}_runs_{num_runs}_cont_{continuous_pct}_sparse_{sparsity}_include_y_{include_y}/'.

4. **Per Run Loop** (tqdm over num_runs):
   - Subdir 'run_{run}'.
   - Generate complete data: generate_data(n, p, continuous_pct, integer_pct, sparsity, include_interactions, include_nonlinear, include_splines, seed + run). Outputs DataFrame ($X_1$-Xp, y), covariates, $\boldsymbol{\beta}$. Save 'complete_data.csv'.
   - Define missingness: define_missingness_patterns(data, seed + run). Returns dict (e.g., 'mcar', 'mar', 'mnar' with Mmis, vars, output).
   - Apply missingness: apply_missingness(data, pattern['Mmis'], col_miss=['$X_1$', '$X_2$'], vars=pattern['vars'], output_file). Save incomplete datasets.
   - Impute: For each dataset, call impute_datasets(incomplete, complete for ref, col_miss, include_y, seed + run). Extend to impute_dl for DL methods (e.g., train AE on observed data, optionally including $y$ as input/feature). Generate multiple imputations. Save '{dataset_name}_{method_type}_{method}_imputed_{idx}.csv'.
   - Evaluate: evaluate_all_imputations(complete, imputed_datasets, output_dir). Computes metrics, stratified by include_y. Returns results DataFrame.

5. **Aggregation**: Concat results_all across runs; save 'results_all_runs.csv'. Average by ['missingness', 'method_type', 'method', 'include_y', 'y']; save 'results_averaged.csv'.

Customize via run_simulation(num_runs=1, n=500, include_y=True). Extend DL with review-inspired architectures (e.g., GAIN for GAN-based). Run with varying include_y to test impact.