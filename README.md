# Project Description
The Include y in Multivariate Value Imputation (MVI) project is designed to simulate and analyze imputation methods for multivariate data with missing values. This project focuses on generating synthetic datasets, running simulations to evaluate imputation techniques, and comparing their performance. It is intended for research purposes, particularly in the context of Duke-NUS Summer Research, to improve data imputation strategies in statistical and machine learning applications.

# Project Objectives
- Simulate multivariate datasets with controlled missingness patterns.
- Implement and test various imputation methods (e.g., mean imputation, regression-based imputation).
- Compare the effectiveness of different imputation techniques using metrics like accuracy and error rates.
- Provide a reproducible framework for testing and analyzing imputation methods.

# Requirements and Setup
For a complete setup on a fresh system, please refer to the SETUP.md file included in this directory. This document provides detailed instructions for:
- Installing Anaconda and creating a Conda environment.
- Installing dependencies listed in requirements.txt.
- Configuring the project structure and files.
- Running the simulation and analysis scripts.

Ensure you have generated requirements.txt from your existing virtual environment (e.g., using pip freeze > requirements.txt) and placed it in the project root directory.

# Usage Example
To run the simulation with a sample configuration:
1. Activate your Conda environment:
   conda activate my_env
2. Execute the simulation script with default parameters (adjust as needed):
   & your_path_python/python.exe "your_path_to_root_directory/run_simulation.py"
   - Inside run_simulation.py, you can modify parameters such as sample size, missingness percentage, or number of runs.
3. Perform analysis after simulation:
   & your_path_python/python.exe "your_path_to_root_directory/src/analysis/compare_imputation_methods.py"

Note: Replace your_path_python with the full path to your Python executable (e.g., H:/Anaconda/envs/my_env/python.exe) and your_path_to_root_directory with the full path to your project root (e.g., H:\MAS_Y1\Duke-NUS Summer Research\include_y_python).

# Contributing
Feel free to contribute by adding new imputation methods, improving simulations, or enhancing the analysis scripts. Please follow the project structure and update requirements.txt and SETUP.md as needed.