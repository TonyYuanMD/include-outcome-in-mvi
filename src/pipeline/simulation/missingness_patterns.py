"""Missingness pattern classes for simulation studies."""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from numpy.random import default_rng

class MissingnessPattern(ABC):
    """Abstract base class for missingness patterns.
    
    All missingness patterns must implement:
    - apply(data, seed=123): Apply missingness to data
    - name: Property for descriptive name
    """
    
    @abstractmethod
    def apply(self, data, seed=123):
        """Apply missingness to the data.
        
        Parameters:
        - data: Input DataFrame
        - seed: Random seed
        
        Returns:
        - dat_miss: DataFrame with missing values
        """
        pass
    
    @property
    @abstractmethod
    def name(self):
        """Return descriptive name of the pattern."""
        pass

class MCARPattern(MissingnessPattern):
    def apply(self, data, seed=123):
        rng = default_rng(seed)
        col_miss = ['X1', 'X2']
        dat_miss = data.copy()
        for var in col_miss:
            probs = np.full(len(dat_miss), 0.2)  # 20% missing
            dat_miss[var] = dat_miss[var].where(rng.uniform(size=len(dat_miss)) >= probs, np.nan)
        return dat_miss
    
    @property
    def name(self):
        return 'mcar'

class MARPattern(MissingnessPattern):
    def apply(self, data, seed=123):
        rng = default_rng(seed)
        col_miss = ['X1', 'X2']
        vars = [col for col in data.columns if col not in ['y', 'y_score']] + ['y', 'y_score']
        dat_miss = data.copy()
        Mmis = pd.DataFrame(0.0, index=col_miss, columns=['Intercept'] + vars)
        Mmis.loc['X1', ['Intercept', 'X3', 'X4']] = [-1.5, 0.5, 0.5]
        Mmis.loc['X2', ['Intercept', 'X3', 'X4']] = [-1.5, 0.5, 0.5]
        for var in col_miss:
            coef = Mmis.loc[var].values
            design_matrix = np.column_stack([np.ones(len(dat_miss))] + [dat_miss[v] for v in vars])
            probs = 1 / (1 + np.exp(-design_matrix @ coef))
            dat_miss[var] = dat_miss[var].where(rng.uniform(size=len(dat_miss)) >= probs, np.nan)
        return dat_miss
    
    @property
    def name(self):
        return 'mar'

class MARType2YPattern(MissingnessPattern):
    def apply(self, data, seed=123):
        rng = default_rng(seed)
        col_miss = ['X1', 'X2']
        vars = [col for col in data.columns if col not in ['y', 'y_score']] + ['y', 'y_score']
        dat_miss = data.copy()
        Mmis = pd.DataFrame(0.0, index=col_miss, columns=['Intercept'] + vars)
        Mmis.loc['X1', ['Intercept', 'y']] = [-1.5, 0.5]
        Mmis.loc['X2', ['Intercept', 'y']] = [-1.5, 0.5]
        for var in col_miss:
            coef = Mmis.loc[var].values
            design_matrix = np.column_stack([np.ones(len(dat_miss))] + [dat_miss[v] for v in vars])
            probs = 1 / (1 + np.exp(-design_matrix @ coef))
            dat_miss[var] = dat_miss[var].where(rng.uniform(size=len(dat_miss)) >= probs, np.nan)
        return dat_miss
    
    @property
    def name(self):
        return 'mar_type2y'

class MARType2ScorePattern(MissingnessPattern):
    def apply(self, data, seed=123):
        rng = default_rng(seed)
        col_miss = ['X1', 'X2']
        vars = [col for col in data.columns if col not in ['y', 'y_score']] + ['y', 'y_score']
        dat_miss = data.copy()
        Mmis = pd.DataFrame(0.0, index=col_miss, columns=['Intercept'] + vars)
        Mmis.loc['X1', ['Intercept', 'y_score']] = [-1.5, 0.5]
        Mmis.loc['X2', ['Intercept', 'y_score']] = [-1.5, 0.5]
        for var in col_miss:
            coef = Mmis.loc[var].values
            design_matrix = np.column_stack([np.ones(len(dat_miss))] + [dat_miss[v] for v in vars])
            probs = 1 / (1 + np.exp(-design_matrix @ coef))
            dat_miss[var] = dat_miss[var].where(rng.uniform(size=len(dat_miss)) >= probs, np.nan)
        return dat_miss
    
    @property
    def name(self):
        return 'mar_type2score'

class MNARPattern(MissingnessPattern):
    def apply(self, data, seed=123):
        rng = default_rng(seed)
        col_miss = ['X1', 'X2']
        vars = [col for col in data.columns if col not in ['y', 'y_score']] + ['y', 'y_score']
        dat_miss = data.copy()
        Mmis = pd.DataFrame(0.0, index=col_miss, columns=['Intercept'] + vars)
        Mmis.loc['X1', ['Intercept', 'X1']] = [-1.5, 0.03]
        Mmis.loc['X2', ['Intercept', 'X2']] = [-1.5, 0.05]
        for var in col_miss:
            coef = Mmis.loc[var].values
            design_matrix = np.column_stack([np.ones(len(dat_miss))] + [dat_miss[v] for v in vars])
            probs = 1 / (1 + np.exp(-design_matrix @ coef))
            dat_miss[var] = dat_miss[var].where(rng.uniform(size=len(dat_miss)) >= probs, np.nan)
        return dat_miss
    
    @property
    def name(self):
        return 'mnar'

class MARThresholdPattern(MissingnessPattern):
    def apply(self, data, seed=123):
        rng = default_rng(seed)
        col_miss = ['X1', 'X2']
        vars = [col for col in data.columns if col not in ['y', 'y_score']] + ['y', 'y_score']
        dat_miss = data.copy()
        Mmis = pd.DataFrame(0.0, index=col_miss, columns=['Intercept'] + vars)
        Mmis.loc['X1', ['Intercept', 'X3']] = [-1.5, 0.5]
        Mmis.loc['X2', ['Intercept', 'X4']] = [-1.5, 0.5]
        for var in col_miss:
            coef = Mmis.loc[var].values
            design_matrix = np.column_stack([np.ones(len(dat_miss))] + [dat_miss[v] for v in vars])
            probs = 1 / (1 + np.exp(-design_matrix @ coef))
            dat_miss[var] = dat_miss[var].where((dat_miss[var] > 0) | (rng.uniform(size=len(dat_miss)) >= probs), np.nan)
        return dat_miss
    
    @property
    def name(self):
        return 'mar_threshold'