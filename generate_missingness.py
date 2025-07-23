import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(123)

# Step 1: Load dataset
dat = pd.read_csv('syn_data/original_data.csv')

# Step 2: Define Mmis matrices (inspired by Table 2)
vars = ['age', 'surgery', 'collapse', 'respiratory_rate', 'chest_xray', 'PE']
col_miss = ['age', 'respiratory_rate']  # Variables to introduce missingness

# MCAR: Random missingness (~20%)
Mmis_mcar = pd.DataFrame(0, index=col_miss, columns=['Intercept'] + vars)
Mmis_mcar.loc['age', 'Intercept'] = np.log(0.2 / 0.8)  # ~20% missing
Mmis_mcar.loc['respiratory_rate', 'Intercept'] = np.log(0.2 / 0.8)

# MAR: Missingness depends on observed variables (surgery, collapse)
Mmis_mar = pd.DataFrame(0, index=col_miss, columns=['Intercept'] + vars)
Mmis_mar.loc['age', ['Intercept', 'surgery', 'collapse']] = [-1.5, 0.5, 0.5]
Mmis_mar.loc['respiratory_rate', ['Intercept', 'surgery', 'collapse']] = [-1.5, 0.5, 0.5]

# MAR Type 2: Missingness depends on outcome (PE)
Mmis_mar_type2 = pd.DataFrame(0, index=col_miss, columns=['Intercept'] + vars)
Mmis_mar_type2.loc['age', ['Intercept', 'PE']] = [-1.5, 1.5]
Mmis_mar_type2.loc['respiratory_rate', ['Intercept', 'PE']] = [-1.5, 1.5]

# MNAR: Missingness depends on the variable itself
Mmis_mnar = pd.DataFrame(0, index=col_miss, columns=['Intercept'] + vars)
Mmis_mnar.loc['age', ['Intercept', 'age']] = [-1.5, 0.03]  # Scaled for age units
Mmis_mnar.loc['respiratory_rate', ['Intercept', 'respiratory_rate']] = [-1.5, 0.05]  # Scaled for resp rate

# Threshold Effect: respiratory_rate missingness increases when age > 65
threshold_age = 65
dat['age_above_threshold'] = (dat['age'] > threshold_age).astype(int)
Mmis_mar_threshold = Mmis_mar.copy()
Mmis_mar_threshold.loc['respiratory_rate', 'Intercept'] = -2.0  # Lower baseline
Mmis_mar_threshold['age_above_threshold'] = 0
Mmis_mar_threshold.loc['respiratory_rate', 'age_above_threshold'] = 1.5  # Increase when age > 65

# Step 3: Function to apply missingness
def apply_missingness(data, Mmis, col_miss, vars):
    dat_miss = data.copy()
    for var in col_miss:
        coef = Mmis.loc[var].values
        design_matrix = np.column_stack([np.ones(len(data))] + [data[v] for v in vars])
        probs = 1 / (1 + np.exp(-design_matrix @ coef))
        dat_miss[var] = dat_miss[var].where(np.random.uniform(size=len(data)) >= probs, np.nan)
    return dat_miss

# Step 4: Generate datasets with missingness
vars_all = vars + ['age_above_threshold']
dat_mcar = apply_missingness(dat, Mmis_mcar, col_miss, vars)
dat_mar = apply_missingness(dat, Mmis_mar, col_miss, vars)
dat_mar_type2 = apply_missingness(dat, Mmis_mar_type2, col_miss, vars)
dat_mnar = apply_missingness(dat, Mmis_mnar, col_miss, vars)
dat_mar_threshold = apply_missingness(dat, Mmis_mar_threshold, col_miss, vars_all)

# Step 5: Validate missingness rates
print("MCAR Missingness Rates:")
print(dat_mcar[col_miss].isna().mean())
print("\nMAR Missingness Rates:")
print(dat_mar[col_miss].isna().mean())
print("\nMAR Type 2 Missingness Rates:")
print(dat_mar_type2[col_miss].isna().mean())
print("\nMNAR Missingness Rates:")
print(dat_mnar[col_miss].isna().mean())
print("\nMAR with Threshold Missingness Rates:")
print(dat_mar_threshold[col_miss].isna().mean())

# Check threshold effect
print(f"\nRespiratory Rate Missingness Rate when Age > {threshold_age}:")
print(dat_mar_threshold.loc[dat['age'] > threshold_age, 'respiratory_rate'].isna().mean())
print(f"Respiratory Rate Missingness Rate when Age <= {threshold_age}:")
print(dat_mar_threshold.loc[dat['age'] <= threshold_age, 'respiratory_rate'].isna().mean())

# Step 6: Save datasets
dat_mcar.to_csv('syn_data/dat_mcar.csv', index=False)
dat_mar.to_csv('syn_data/dat_mar.csv', index=False)
dat_mar_type2.to_csv('syn_data/dat_mar_type2.csv', index=False)
dat_mnar.to_csv('syn_data/dat_mnar.csv', index=False)
dat_mar_threshold.to_csv('syn_data/dat_mar_threshold.csv', index=False)

# Step 7: Documentation
"""
Missingness Patterns (inspired by PIIS Table 2):
- MCAR: Random missingness (~20%) for age, respiratory_rate.
- MAR: Age, respiratory_rate missingness depends on surgery and collapse (e.g., surgical patients or those with collapse more likely to have missing data).
- MAR Type 2: Age, respiratory_rate missingness depends on PE (e.g., patients with PE more likely to have missing data).
- MNAR: Age, respiratory_rate missingness depends on their own values (e.g., older age or higher respiratory rates more likely missing).
- MAR with Threshold: Respiratory_rate missingness increases when age > 65 (e.g., older patients have more missing respiratory data).
"""