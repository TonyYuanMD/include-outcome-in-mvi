import numpy as np
import pandas as pd

def apply_missingness(data, Mmis, col_miss, vars, output_file):
    """
    Apply missingness to specified columns based on Mmis matrix.
    
    Parameters:
    - data: Input DataFrame
    - Mmis: Missingness matrix
    - col_miss: Columns to introduce missingness
    - vars: Predictor variables for missingness
    - output_file: Path to save output
    
    Returns:
    - dat_miss: DataFrame with missing values
    """
    dat_miss = data.copy()
    for var in col_miss:
        coef = Mmis.loc[var].values
        design_matrix = np.column_stack([np.ones(len(dat_miss))] + [dat_miss[v] for v in vars])
        probs = 1 / (1 + np.exp(-design_matrix @ coef))
        dat_miss[var] = dat_miss[var].where(np.random.uniform(size=len(dat_miss)) >= probs, np.nan)
    dat_miss.to_csv(output_file, index=False)
    return dat_miss

def define_missingness_patterns(data, col_miss=['X1', 'X2'], seed=123):
    """
    Define missingness patterns and their coefficients.
    
    Parameters:
    - data: Input DataFrame
    - col_miss: Columns for missingness
    - seed: Random seed
    
    Returns:
    - patterns: Dictionary of missingness patterns
    """
    np.random.seed(seed)
    covariates = [col for col in data.columns if col not in ['y', 'y_score']]
    vars = covariates + ['y', 'y_score']
    
    patterns = {
        'mcar': {
            'Mmis': pd.DataFrame(0, index=col_miss, columns=['Intercept'] + vars),
            'vars': vars,
            'output': 'dat_mcar.csv'
        },
        'mar': {
            'Mmis': pd.DataFrame(0, index=col_miss, columns=['Intercept'] + vars),
            'vars': vars,
            'output': 'dat_mar.csv'
        },
        'mar_type2_y': {
            'Mmis': pd.DataFrame(0, index=col_miss, columns=['Intercept'] + vars),
            'vars': vars,
            'output': 'dat_mar_type2_y.csv'
        },
        'mar_type2_score': {
            'Mmis': pd.DataFrame(0, index=col_miss, columns=['Intercept'] + vars),
            'vars': vars,
            'output': 'dat_mar_type2_score.csv'
        },
        'mnar': {
            'Mmis': pd.DataFrame(0, index=col_miss, columns=['Intercept'] + vars),
            'vars': vars,
            'output': 'dat_mnar.csv'
        },
        'mar_threshold': {
            'Mmis': pd.DataFrame(0, index=col_miss, columns=['Intercept'] + vars),
            'vars': vars,
            'output': 'dat_mar_threshold.csv'
        }
    }
    
    patterns['mcar']['Mmis'].loc['X1', 'Intercept'] = np.log(0.2 / 0.8)
    patterns['mcar']['Mmis'].loc['X2', 'Intercept'] = np.log(0.2 / 0.8)
    patterns['mar']['Mmis'].loc['X1', ['Intercept', 'X3', 'X4']] = [-1.5, 0.5, 0.5]
    patterns['mar']['Mmis'].loc['X2', ['Intercept', 'X3', 'X4']] = [-1.5, 0.5, 0.5]
    patterns['mar_type2_y']['Mmis'].loc['X1', ['Intercept', 'y']] = [-1.5, 1.5]
    patterns['mar_type2_y']['Mmis'].loc['X2', ['Intercept', 'y']] = [-1.5, 1.5]
    patterns['mar_type2_score']['Mmis'].loc['X1', ['Intercept', 'y_score']] = [-1.5, 0.3]
    patterns['mar_type2_score']['Mmis'].loc['X2', ['Intercept', 'y_score']] = [-1.5, 0.3]
    patterns['mnar']['Mmis'].loc['X1', ['Intercept', 'X1']] = [-1.5, 0.03]
    patterns['mnar']['Mmis'].loc['X2', ['Intercept', 'X2']] = [-1.5, 0.05]
    patterns['mar_threshold']['Mmis'].loc['X1', ['Intercept', 'X3', 'X4']] = [-1.5, 0.5, 0.5]
    patterns['mar_threshold']['Mmis'].loc['X2', ['Intercept', 'X3', 'X4', 'X1_above_threshold']] = [-2.0, 0.5, 0.5, 1.5]
    
    return patterns

# Documentation
"""
Functions:
- apply_missingness: Applies missingness to specified columns using Mmis matrix
- define_missingness_patterns: Defines missingness patterns (MCAR, MAR, etc.)
- Description: Generates datasets with missing values in X1, X2
"""