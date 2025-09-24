# Setup Instructions for Include y in Multivariate Value Imputation (MVI) Project

This document provides a step-by-step guide to set up the Include y in Multivariate Value Imputation (MVI) project on a fresh Windows system with Anaconda. The instructions assume you are working in the project root directory and using the generated requirements.txt file from your existing virtual environment.

## Prerequisites
- Anaconda Distribution[](https://www.anaconda.com/products/distribution) installed on your system.
- Write access to the project directory.

## Step-by-Step Setup

### 1. Install Anaconda
- Download and install Anaconda from https://www.anaconda.com/products/distribution if not already installed.
- During installation, check "Add Anaconda to my PATH environment variable" (optional but simplifies usage).
- Verify installation by opening PowerShell and running: `conda --version`

### 2. Create a New Conda Environment
- Navigate to the project directory:
- Create a new Conda environment with Python 3.10 (matching your current setup):
  `conda create -n my_env python=3.10 -y`
- Activate the environment:
  `conda activate my_env`

### 3. Install Dependencies
- Ensure `requirements.txt` is in the current directory.
- Install all dependencies listed in `requirements.txt`: `pip install -r requirements.txt`
- Verify installation by checking versions (optional): `python -c "import numpy, pandas, pytest; print(numpy.__version__, pandas.__version__, pytest.__version__)"`

### 4. Project Structure and File Placement
Ensure the following structure is in place (create directories/files as needed):

|-run_simulation.py  # Main simulation script at root
|-src
|  |-pipeline
|  |-analysis
|-tests
|  |-test_run_simulation.py  # Test file
|-artifacts
|-data
|-result

- Place `run_simulation.py` at the root if not already there. If it depends on modules in `src/` (e.g., `generate_data` from `src.pipeline`), ensure those files are present and correctly imported.



### 5. Run the Simulation
- Test the setup by running the simulation manually (adjust parameters as needed):
  `& your_path_python/python.exe "your_path_to_root_directory/run_simulation.py"`
- Inside the `run_simulation.py`, you can change the parameter values.

### 6. Perform Analysis
- Run analysis after simualtion manually:
  `& your_path_python/python.exe "your_path_to_root_directory/src/analysis/compare_imputation_methods.py"`

