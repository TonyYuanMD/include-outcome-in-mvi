# Reflection on Simulation Study Design and Implementation

## Design Justification and Conditions

### Why These DGPs?

The core purpose of this simulation is to evaluate **predictive utility** under conditions that challenge simple imputation assumptions. The design choices for the Data Generating Processes (DGPs) were made to control complexity and variability:

1. **Mixed Covariate Types:** The inclusion of continuous, integer, and binary variables reflects real-world data, where imputation models must handle diverse distributions.
    
2. **Model Complexity Flags (**$\text{Interactions}$**,** $\text{Nonlinear}$**,** $\text{Splines}$**):** These flags allow us to vary the true relationship between $X$ and $Y$. By training a prediction model (Linear/Logistic Regression) on the imputed data, we test if the imputation method can recover complex underlying structures or if it only works when relationships are purely linear.
    
3. **Sparsity:** Introducing zero coefficients ensures that the signal-to-noise ratio is manageable, preventing the model from becoming overly complex and unstable, which is critical for robust testing.
    

### Why These Missingness Patterns?

The six chosen missingness patterns ($\text{MCAR}$, $\text{MAR}$, $\text{MNAR}$, etc.) are non-negotiable for an imputation study as they cover the theoretical mechanisms:

- **MCAR (Missing Completely At Random):** The theoretical baseline where all methods are expected to perform well.
    
- **MAR (Missing At Random):** The most common scenario where missingness depends on _observed_ data ($\text{X3}$, $\text{X4}$).
    
- **Outcome-Dependent MAR (**$\text{MARType2Y}$**,** $\text{MARType2Score}$**):** A crucial test case for this study. It mimics situations (common in clinical trials) where the propensity to miss data depends on the outcome ($\text{Y}$) itself.
    
- **MNAR (Missing Not At Random):** The hardest case, where missingness depends on the value of the missing variable itself, setting a lower bound on imputation performance.
    

## Ensuring Fairness and Confidence

### Fairness and Unbiasedness

Fairness was ensured primarily through rigorous control of randomness and the **Utility-Based Evaluation** design:

1. **Utility Evaluation:** We abandoned the flawed comparison of $Y_{\text{true}}$ to $Y_{\text{imputed}}$ and adopted the gold standard: training a model on the imputed data and evaluating its performance on a **new, independent, complete test set**. This ensures results are unbiased toward the downstream task.
    
2. **Two-Tiered Randomness:**
    
    - **Simulation Uncertainty (**$\text{STD\_Runs}$**):** Running the simulation multiple times ($\text{num\_runs}$) with different random seeds for data generation prevents results from being idiosyncratic to a single, lucky dataset.
        
    - **Imputation Uncertainty (**$\text{STD}$**):** Using multiple imputation ($\text{n\_imputations} > 1$) for stochastic methods (like MICE) measures how stable a method is when the imputation itself is randomized.
        

### Confidence in Results and Potential Undermining Factors

I am highly confident in the **relative comparison** between the imputation methods. The utility-based evaluation ensures that if Method A scores lower Log Loss than Method B, Method A truly provides a better dataset for prediction.

However, confidence could be undermined by:

1. **Small Sample Size (**$\text{n=50}$**):** The initial small sample size leads to highly unstable prediction models (evidenced by the negative $\text{R}^2$ and high $\text{Log Loss}$), inflating the variance of the results. While the $\text{STD\_Runs}$ captures this, the absolute results are poor.
    
2. **Model Mismatch:** We used simple Linear/Logistic Regression for the downstream task. If a more complex model (like a Gradient Boosting Machine) were used, the results might favor different imputation methods (e.g., those that better handle non-linearities, like MissForest).
    
3. **Single Missingness Level:** We tested a fixed $\text{20\%}$ missing rate. The performance of deep learning methods (like GAIN) might change drastically at higher or lower missing rates.
    

## Limitations and Future Work

### Scenarios Not Included

Two crucial scenarios were omitted due to scope and complexity:

1. **High-Dimensional Data (**$\mathbf{p \gg n}$**):** We primarily focused on $n \ge p$. In modern contexts, $p$ often exceeds $n$, which would expose limitations in standard regression-based imputation (MICE) and highlight the need for regularized imputation models.
    
2. **Non-Continuous Missingness:** We did not introduce missingness into categorical or binary covariates, requiring separate, specialized imputation models (e.g., Logistic Regression within MICE) which would significantly increase implementation time.
    

These scenarios matter because they directly impact the scalability and practical application of the imputation methods.

### How Results Inform Practice or Theory

The results are valuable for both:

- **Practice:** They provide clear evidence against using simplistic methods (Mean/Single Imputation) when data is MAR or MNAR, particularly when the downstream goal is prediction. They empirically show that more sophisticated, model-based methods (MICE, GAIN) are necessary.
    
- **Theory:** The comparison of $\text{with\_y}$ vs. $\text{without}$ directly tests the theoretical concept of **"Using the outcome in imputation."** If $\text{MICE\_with\_y}$ performs significantly better than $\text{MICE\_without}$, it supports the use of outcome data to maximize predictive power, even if it risks introducing bias.
    

## Implementation Challenges

The most challenging aspect of the implementation was designing the **robust flow of randomness and evaluation**:

- **RNG Management:** Ensuring that the $\text{n\_imputations}$ randomness was correctly isolated from the $\text{num\_runs}$ randomness (using `numpy.random.default_rng` spawning) was complex. A single error in seed management would have invalidated the entire $\text{STD}$ calculation.
    
- **Refactoring the Evaluator:** The shift from the initially flawed $Y_{true}$ vs $Y_{imputed}$ comparison to the final **utility-based evaluation** required restructuring `evaluator.py` and updating the data passing logic in `simulator.py`, ensuring all metrics were correctly prefixed and aggregated.