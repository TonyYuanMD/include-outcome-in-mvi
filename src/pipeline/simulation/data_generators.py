"""Data generation for simulation studies."""

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from numpy.random import default_rng

def generate_data(n=1000, p=5, continuous_pct=0.4, integer_pct=0.4, sparsity=0.3,
                  include_interactions=True, include_nonlinear=True, include_splines=True,
                  rng=None):
    """
    Generate synthetic data with configurable covariate types and response variables.
    
    Parameters:
    - n: Sample size
    - p: Number of base covariates
    - continuous_pct: Proportion of continuous covariates
    - integer_pct: Proportion of integer covariates
    - sparsity: Proportion of zero coefficients
    - include_interactions: Include pairwise interaction terms
    - include_nonlinear: Include sin, cos transformations
    - include_splines: Include spline basis expansion
    - seed: Random seed
    
    Returns:
    - data: DataFrame with covariates, X1_above_threshold, y, y_score
    - covariates: List of covariate names
    - beta: Coefficient array
    """
    if rng is None:
        rng = default_rng(123)
    
    # Generate covariates
    num_continuous = int(p * continuous_pct)
    num_integer = int(p * integer_pct)
    num_binary = p - num_continuous - num_integer
    data = {}
    covariates = []
    for i in range(p):
        name = f'X{i+1}'
        z = rng.normal(0, 1, n)
        if i < num_continuous:
            data[name] = z
        elif i < num_continuous + num_integer:
            data[name] = np.round(z).astype(int)
        else:
            data[name] = (z > 0).astype(int)
        covariates.append(name)
    
    # Add X1_above_threshold
    data['X1_above_threshold'] = (data['X1'] > 0).astype(int)
    covariates.append('X1_above_threshold')
    
    # Add interaction terms
    if include_interactions:
        for i in range(p):
            for j in range(i + 1, p):
                name = f'{covariates[i]}_{covariates[j]}'
                data[name] = data[covariates[i]] * data[covariates[j]]
                covariates.append(name)
    
    # Add nonlinear terms
    if include_nonlinear:
        for i in range(num_continuous):
            name = covariates[i]
            data[f'{name}_sin'] = np.sin(data[name])
            data[f'{name}_cos'] = np.cos(data[name])
            covariates.extend([f'{name}_sin', f'{name}_cos'])
    
    # Add spline basis expansion
    if include_splines:
        for i in range(num_continuous):
            name = covariates[i]
            x = data[name]
            knots = np.percentile(x, [25, 50, 75])
            degree = 3
            for j in range(degree + 1):
                coeffs = np.zeros(degree + 1)
                coeffs[j] = 1
                spline = BSpline(knots, coeffs, degree)(x)
                data[f'{name}_spline{j+1}'] = spline
                covariates.append(f'{name}_spline{j+1}')
    
    # Generate coefficients with sparsity
    num_predictors = len(covariates)
    num_nonzero = int(num_predictors * (1 - sparsity))
    nonzero_indices = rng.choice(num_predictors, num_nonzero, replace=False)
    beta = np.zeros(num_predictors + 1)
    beta[0] = rng.normal(0, 1)
    beta[nonzero_indices + 1] = rng.normal(0, 1, num_nonzero)
    
    # Generate outcomes
    design_matrix = np.column_stack([np.ones(n)] + [data[cov] for cov in covariates])
    logits = design_matrix @ beta
    probs_y = 1 / (1 + np.exp(-logits))
    data['y'] = rng.binomial(1, probs_y)
    data['y_score'] = design_matrix @ beta + rng.normal(0, 1, n)
    
    return pd.DataFrame(data), covariates, beta