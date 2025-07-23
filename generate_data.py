import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(123)

# Step 1: Define parameters from Table 1
n = 1000  # Sample size
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
- age: Continuous, mean=56.1, SD=17.5 (years)
- surgery: Binary, prevalence=0.214 (surgery within past 3 months)
- collapse: Binary, prevalence=0.074 (collapse event)
- respiratory_rate: Integer, mean=19.4, SD=6.7 (breaths per minute, rounded)
- chest_xray: Binary, prevalence=0.409 (abnormal chest X-ray)
- PE: Binary outcome (Pulmonary Embolism), generated using logistic regression
  with true coefficients from Table 1 (Intercept=-2.95, age=0.017, surgery=0.51,
  collapse=1.35, respiratory_rate=0.057, chest_xray=0.81)
"""