import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(123)

# Step 1: Define parameters from Table 1
n = 1000  # Sample size
covariates = {
    'age': {'mean': 60, 'sd': 15, 'type': 'continuous'},
    'surgery': {'prevalence': 0.20, 'type': 'binary'},
    'collapse': {'prevalence': 0.20, 'type': 'binary'},
    'respiratory_rate': {'mean': 24, 'sd': 8, 'type': 'integer'},
    'chest_xray': {'prevalence': 0.50, 'type': 'binary'}
}
true_beta = {
    'Intercept': -3.5,
    'age': 0.03,
    'surgery': 0.9,
    'collapse': 0.9,
    'respiratory_rate': 0.05,
    'chest_xray': 0.7
}

# Step 2: Generate covariates
data = {}
# Age (continuous)
data['age'] = np.random.normal(loc=covariates['age']['mean'], 
                              scale=covariates['age']['sd'], size=n)
# Surgery (binary)
data['surgery'] = np.random.binomial(1, covariates['surgery']['prevalence'], size=n)
# Collapse (binary)
data['collapse'] = np.random.binomial(1, covariates['collapse']['prevalence'], size=n)
# Respiratory rate (continuous then rounded to integer)
data['respiratory_rate'] = np.round(np.random.normal(loc=covariates['respiratory_rate']['mean'], 
                                                    scale=covariates['respiratory_rate']['sd'], 
                                                    size=n)).astype(int)
# Chest X-ray (binary)
data['chest_xray'] = np.random.binomial(1, covariates['chest_xray']['prevalence'], size=n)

# Step 3: Generate outcome (PE) using logistic regression
design_matrix = np.column_stack([
    np.ones(n),  # Intercept
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

# Step 4: Create DataFrame
dat = pd.DataFrame(data)

# Step 5: Save dataset
dat.to_csv(path_or_buf="syn_data/original_data.csv", index=False)

# Step 6: Documentation
"""
Dataset Description:
- age: Continuous, mean=60, SD=15 (years)
- surgery: Binary, prevalence=0.20 (surgery within past 3 months)
- collapse: Binary, prevalence=0.20 (collapse event)
- respiratory_rate: Integer, mean=24, SD=8 (breaths per minute, rounded)
- chest_xray: Binary, prevalence=0.50 (abnormal chest X-ray)
- PE: Binary outcome (Pulmonary Embolism), generated using logistic regression
  with true coefficients from Table 1 (Intercept=-3.5, age=0.03, surgery=0.9,
  collapse=0.9, respiratory_rate=0.05, chest_xray=0.7)
"""