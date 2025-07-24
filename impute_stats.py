import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(123)

# Step 1: Load datasets
datasets = {
    'mcar': pd.read_csv('syn_data/dat_mcar.csv'),
    'mar': pd.read_csv('syn_data/dat_mar.csv'),
    'mar_type2_pe': pd.read_csv('syn_data/dat_mar_type2_pe.csv'),
    'mar_type2_score': pd.read_csv('syn_data/dat_mar_type2_score.csv'),
    'mnar': pd.read_csv('syn_data/dat_mnar.csv'),
    'mar_threshold': pd.read_csv('syn_data/dat_mar_threshold.csv')
}
original_data = pd.read_csv('syn_data/original_data.csv')

# Step 2: Define imputation functions
def mean_imputation(data):
    dat_imputed = data.copy()
    for col in ['age', 'respiratory_rate']:
        dat_imputed[col] = dat_imputed[col].fillna(dat_imputed[col].mean())
    return [dat_imputed]

def single_imputation(data, outcome='PE'):
    dat_imputed = data.copy()
    predictors = ['surgery', 'collapse', 'chest_xray', outcome]
    for col in ['age', 'respiratory_rate']:
        mask = ~data[col].isna()
        X_train = data.loc[mask, predictors]
        y_train = data.loc[mask, col]
        model = LinearRegression().fit(X_train, y_train)
        mask_missing = data[col].isna()
        X_missing = data.loc[mask_missing, predictors]
        if len(X_missing) > 0:
            dat_imputed.loc[mask_missing, col] = model.predict(X_missing)
    return [dat_imputed]

def mice_imputation(data, outcome='PE', n_imputations=5):
    dat_imputed_list = []
    predictors = ['age', 'surgery', 'collapse', 'respiratory_rate', 'chest_xray', outcome]
    for i in range(n_imputations):
        imputer = IterativeImputer(max_iter=10, random_state=123 + i)
        dat_imputed = data.copy()
        dat_imputed[predictors] = imputer.fit_transform(data[predictors])
        dat_imputed_list.append(dat_imputed)
    return dat_imputed_list

# Step 3: Perform imputation with train/test split
imputed_datasets = {}
for dataset_name, data in datasets.items():
    imputed_datasets[dataset_name] = {}
    # Train/test split (80/20)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=123)
    # Impute for both outcomes
    for outcome in ['PE', 'PE_score']:
        # Mean imputation
        imputed_datasets[dataset_name][f'mean_{outcome}'] = {
            'train': mean_imputation(train_data),
            'test': mean_imputation(test_data),
            'full': mean_imputation(data)
        }
        # Single imputation
        imputed_datasets[dataset_name][f'single_with_{outcome}'] = {
            'train': single_imputation(train_data, outcome=outcome),
            'test': single_imputation(test_data, outcome=outcome),
            'full': single_imputation(data, outcome=outcome)
        }
        imputed_datasets[dataset_name][f'single_without_{outcome}'] = {
            'train': single_imputation(train_data.drop(columns=[outcome]), outcome=outcome),
            'test': single_imputation(test_data.drop(columns=[outcome]), outcome=outcome),
            'full': single_imputation(data.drop(columns=[outcome]), outcome=outcome)
        }
        # MICE imputation
        imputed_datasets[dataset_name][f'mice_with_{outcome}'] = {
            'train': mice_imputation(train_data, outcome=outcome),
            'test': mice_imputation(test_data, outcome=outcome),
            'full': mice_imputation(data, outcome=outcome)
        }
        imputed_datasets[dataset_name][f'mice_without_{outcome}'] = {
            'train': mice_imputation(train_data.drop(columns=[outcome]), outcome=outcome),
            'test': mice_imputation(test_data.drop(columns=[outcome]), outcome=outcome),
            'full': mice_imputation(data.drop(columns=[outcome]), outcome=outcome)
        }

# Step 4: Documentation
"""
Statistical Imputation:
- Mean: Fills age, respiratory_rate with means.
- Single: Linear regression with surgery, collapse, chest_xray, optionally PE/PE_score.
- MICE: 5 imputations with linear regression, optionally including PE/PE_score.
- Train/Test Split: 80/20 split on datasets with missingness.
Output: Dictionary of imputed train/test/full datasets (imputed_datasets[dataset_name][method][split]).
"""