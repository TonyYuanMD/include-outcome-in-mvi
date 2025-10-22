# Project Description

The **Include Y in Multivariate Value Imputation (MVI)** project is designed to simulate and analyze imputation methods for multivariate data with missing values. This project focuses on generating synthetic datasets, running simulations to evaluate imputation techniques, and comparing their predictive utility. It is intended for research purposes, particularly in the context of Duke-NUS Summer Research, to improve data imputation strategies in statistical and machine learning applications.

# Project Objectives

- Simulate multivariate datasets with controlled missingness patterns.
    
- Implement and test various imputation methods (from mean imputation to deep learning like GAIN).
    
- **Compare the predictive utility** of different imputation techniques using metrics like Log Loss and $R^2$ on downstream models.
    
- Provide a reproducible framework for testing and analyzing imputation methods.
    

# Requirements and Setup

For a complete setup on a fresh system, please refer to the `SETUP.md` file (not included here). This document provides detailed instructions for:

- Installing dependencies listed in `requirements.txt`.
    
- Configuring the project structure (ensuring `__init__.py` files exist in `src/`, `src/pipeline/`, etc.).
    

# Usage and Pipeline Automation (Using Makefile)

The entire simulation, analysis, and cleaning pipeline is automated using the `Makefile` located in the root directory. This requires having the `make` utility installed (common in Linux/macOS, available via Git Bash on Windows).

|   |   |   |
|---|---|---|
|**Target**|**Command**|**Description**|
|**Complete Run**|`make all`|**Runs the entire pipeline:** Executes tests, then simulation, analysis, and figure generation in sequence.|
|**Simulation**|`make simulate`|Runs `run_simulation.py` to execute the Monte Carlo study and save raw results (`results_all_runs.csv`).|
|**Analysis/Figures**|`make figures`|Runs `compare_imputation_methods.py` to load averaged results, perform meta-averaging, and generate all visualizations (`results/figures/`).|
|**Testing**|`make test`|Runs the test suite using **Pytest** on the `tests/` directory.|
|**Clean**|`make clean`|Removes all generated output files (`results/` directory and `simulation.log.txt`).|

# Testing the Pipeline

To ensure the integrity and stability of the simulation logic:

1. **Navigate to the project root.**
    
2. **Execute Pytest:** `make test`
    

The test suite verifies core mechanisms such as $\text{Log Loss}$ vs. $R^2$ metric selection and the calculation of **Simulation Uncertainty** ($\text{STD}$ across runs).

# Contributing

Feel free to contribute by adding new imputation methods, improving simulation scenarios, or enhancing the analysis scripts. Please follow the project structure and update project documentation as needed.