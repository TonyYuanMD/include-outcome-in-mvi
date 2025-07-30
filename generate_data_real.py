import numpy as np
import pandas as pd

def generate_data_realistic(n=1000, seed=123, output_dir="syn_data"):
    """
    Generate realistic synthetic data mimicking PIIS paper.
    
    Parameters:
    - n: Sample size
    - seed: Random seed
    - output_dir: Directory to save output
    
    Returns:
    - data: DataFrame with covariates, PE, PE_score
    - true_beta: Dictionary of coefficients
    """
    np.random.seed(seed)
    
    # Define parameters from Table 1 (PIIS paper)
    covariates = {
        'age': {'mean': 56.1, 'sd': 17.5, 'type': 'continuous'},
        'surgery': {'prevalence': 0.214, 'type': 'binary'},
        'collapse': {'prevalence': 0.074, 'type': 'binary'},
        'respiratory_rate': {'mean': 19.4, 'sd': 6.7, 'type': 'integer'},
        'chest_xray': {'prevalence': 0.409, 'type': 'binary'}
    }
    true_beta = {
        'Intercept': -2.95,
        'age': 0.017,
        'surgery': 0.51,
        'collapse': 1.35,
        'respiratory_rate': 0.057,
        'chest_xray': 0.81
    }
    
    # Generate covariates
    data = {}
    data['age'] = np.random.normal(loc=covariates['age']['mean'], 
                                  scale=covariates['age']['sd'], size=n)
    data['surgery'] = np.random.binomial(1, covariates['surgery']['prevalence'], size=n)
    data['collapse'] = np.random.binomial(1, covariates['collapse']['prevalence'], size=n)
    data['respiratory_rate'] = np.round(np.random.normal(loc=covariates['respiratory_rate']['mean'], 
                                                        scale=covariates['respiratory_rate']['sd'], 
                                                        size=n)).astype(int)
    data['chest_xray'] = np.random.binomial(1, covariates['chest_xray']['prevalence'], size=n)
    
    # Generate outcomes
    design_matrix = np.column_stack([
        np.ones(n),
        data['age'],
        data['surgery'],
        data['collapse'],
        data['respiratory_rate'],
        data['chest_xray']
    ])
    beta_vector = np.array([true_beta['Intercept'], true_beta['age'], true_beta['surgery'], 
                            true_beta['collapse'], true_beta['respiratory_rate'], true_beta['chest_xray']])
    
    logits = design_matrix @ beta_vector
    probs_pe = 1 / (1 + np.exp(-logits))
    data['PE'] = np.random.binomial(1, probs_pe)
    data['PE_score'] = design_matrix @ beta_vector + np.random.normal(0, 1, n)
    
    # Create DataFrame
    data = pd.DataFrame(data)
    
    return data, true_beta

# Documentation
"""
Function: generate_data_realistic
- Description: Generates realistic synthetic data based on PIIS paper.
- Covariates: age (continuous), surgery (binary), collapse (binary), respiratory_rate (integer), chest_xray (binary)
- Outcomes: PE (binary), PE_score (continuous)
- Returns: DataFrame, true coefficients dictionary
"""