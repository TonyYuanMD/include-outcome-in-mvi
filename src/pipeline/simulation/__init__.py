"""Simulation study framework for missing value imputation (MVI) in healthcare data.

This package provides a modular framework for simulating and evaluating imputation methods under various missingness patterns, inspired by deep learning techniques reviewed in healthcare contexts.

Basic Usage
-----------
>>> from simulation import SimulationStudy
>>> from missingness_patterns import MCARPattern
>>> from imputation_methods import MeanImputation
>>>
>>> study = SimulationStudy(n=500, num_runs=1, seed=123)
>>> missingness = MCARPattern()
>>> imputer = MeanImputation()
>>> results = study.run_scenario(missingness, imputer)
>>> print(results)

Modules
-------
data_generators : Complete data generation
missingness_patterns : Missingness application classes
imputation_methods : Imputation method classes
evaluator : Evaluation metrics
simulator : Study orchestration
main : Example scripts
"""

from .data_generators import generate_data
from .missingness_patterns import (
    MissingnessPattern,
    MCARPattern,
    MARPattern,
    MARType2YPattern,
    MARType2ScorePattern,
    MNARPattern,
    MARThresholdPattern
)
from .imputation_methods import (
    ImputationMethod,
    CompleteData,
    MeanImputation,
    SingleImputation,
    MICEImputation,
    MissForestImputation,
    MLPImputation,
    AutoencoderImputation,
    GAINImputation
)
from .evaluator import evaluate_all_imputations
from .simulator import SimulationStudy

__version__ = '1.0.0'

__all__ = [
    # Data generation
    'generate_data',
    
    # Abstract base classes
    'MissingnessPattern',
    'ImputationMethod',
    
    # Concrete missingness patterns
    'MCARPattern',
    'MARPattern',
    'MARType2YPattern',
    'MARType2ScorePattern',
    'MNARPattern',
    'MARThresholdPattern',
    
    # Concrete imputation methods
    'CompleteData',
    'MeanImputation',
    'SingleImputation',
    'MICEImputation',
    'MissForestImputation',
    'MLPImputation',
    'AutoencoderImputation',
    'GAINImputation',
    
    # Evaluation and simulation
    'evaluate_all_imputations',
    'SimulationStudy',
]