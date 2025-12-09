# Include Y in Multivariate Value Imputation (MVI)

A comprehensive simulation framework for evaluating missing value imputation methods in multivariate data, with a focus on predictive utility assessment.

## Project Description

The **Include Y in Multivariate Value Imputation (MVI)** project is designed to simulate and analyze imputation methods for multivariate data with missing values. This project focuses on generating synthetic datasets, running simulations to evaluate imputation techniques, and comparing their predictive utility. It is intended for research purposes, particularly in the context of Duke-NUS Summer Research, to improve data imputation strategies in statistical and machine learning applications.

## Project Objectives

- Simulate multivariate datasets with controlled missingness patterns (MCAR, MAR, MNAR, etc.)
- Implement and test various imputation methods (from mean imputation to deep learning like GAIN)
- **Compare the predictive utility** of different imputation techniques using metrics like Log Loss and $R^2$ on downstream models
- Provide a reproducible framework for testing and analyzing imputation methods

## Features

- **6 Missingness Patterns**: MCAR, MAR, MAR Type 2 Y, MAR Type 2 Score, MNAR, MAR Threshold
- **8 Imputation Methods**: Complete Data, Mean, Single, MICE, MissForest, MLP, Autoencoder, GAIN
- **Full Factorial Design**: Systematic evaluation across parameter combinations
- **Parallel Execution**: Efficient multiprocessing for large-scale simulations
- **JSON Configuration**: Flexible parameter specification via configuration files
- **Comprehensive Analysis**: Statistical tests and visualizations
- **Regression Testing**: Ensures correctness and reproducibility

## Requirements and Setup

### Prerequisites

- Python 3.10 or higher
- `make` utility (optional, for automated workflows)

### Installation

1. **Clone or navigate to the project directory**

2. **Create a virtual environment** (recommended):
   ```bash
   conda create -n mvi_env python=3.10 -y
   conda activate mvi_env
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Core Dependencies

- `numpy>=1.24.0` - Numerical computing
- `pandas>=1.5.0` - Data manipulation
- `scipy>=1.9.0` - Scientific computing
- `scikit-learn>=1.2.0` - Machine learning
- `torch>=2.0.0` - Deep learning (for MLP, Autoencoder, GAIN)
- `matplotlib>=3.6.0` - Visualization
- `seaborn>=0.12.0` - Statistical visualization
- `tqdm>=4.64.0` - Progress bars

For detailed setup instructions on a fresh system, refer to `SETUP.md`.

## Quick Start

### Demo Script

Run the demo script to see a basic example:

```bash
python main.py
```

This will run a simple simulation with:
- Sample size: 100
- Number of predictors: 5
- MCAR missingness pattern
- Mean imputation method

### Basic Usage

```python
from src.pipeline.simulation import SimulationStudy, MCARPattern, MeanImputation
from numpy.random import default_rng

# Create a simulation study
study = SimulationStudy(n=500, p=5, num_runs=1, seed=123)

# Define missingness pattern and imputation method
pattern = MCARPattern()
method = MeanImputation()

# Run a single scenario
rng = default_rng(123)
results = study.run_scenario(pattern, method, run_rng=rng)
print(results)
```

### Running Full Simulations

#### Using Default Parameters

```bash
python run_simulation.py
```

#### Using JSON Configuration

1. Create a configuration file (see `config_example.json`):
   ```json
   {
       "n": [50, 100],
       "p": [5, 10],
       "num_runs": 2,
       "continuous_pct": [0.4],
       "integer_pct": [0.4],
       "sparsity": [0.3],
       "include_interactions": [false],
       "include_nonlinear": [false],
       "include_splines": [false],
       "seed": 123
   }
   ```

2. Run with configuration:
   ```bash
   python -c "from run_simulation import run_simulation; run_simulation(config_file='config_example.json')"
   ```

## Usage and Pipeline Automation (Using Makefile)

The entire simulation, analysis, and cleaning pipeline is automated using the `Makefile` located in the root directory. This requires having the `make` utility installed (common in Linux/macOS, available via Git Bash on Windows).

### Main Targets

| **Target** | **Command** | **Description** |
|---|---|---|
| **Complete Run** | `make all` | Runs the entire pipeline: tests → simulation → analysis → figures |
| **Simulation** | `make simulate` | Runs `run_simulation.py` to execute the Monte Carlo study |
| **Simulation (JSON)** | `make simulate CONFIG=config.json` | Runs simulation with JSON configuration file |
| **Analysis** | `make analyze` | Processes results and generates summary tables |
| **Figures** | `make figures` | Generates visualization figures |
| **Testing** | `make test` | Runs the test suite using **Pytest** |
| **Clean** | `make clean` | Removes all generated output files |

### Performance & Analysis Targets

| **Target** | **Command** | **Description** |
|---|---|---|
| **Profiling** | `make profile` | Run performance profiling on representative simulation |
| **Complexity** | `make complexity` | Analyze computational complexity (timing vs n, p) |
| **Benchmark** | `make benchmark` | Run timing comparison benchmarks |
| **Stability Check** | `make stability-check` | Check for numerical warnings/convergence issues |
| **Performance Comparison** | `make performance-comparison` | Generate baseline vs optimized visualizations |

### Examples

```bash
# Run complete pipeline
make all

# Run simulation with custom config
make simulate CONFIG=config_test.json

# Run tests only
make test

# Profile performance
make profile

# Clean generated files
make clean
```

## Project Structure

```
include_y_python/
├── main.py                      # Demo script
├── run_simulation.py            # Main simulation script
├── requirements.txt             # Python dependencies
├── Makefile                     # Automation targets
├── config_example.json          # Example configuration
├── src/
│   ├── pipeline/
│   │   └── simulation/
│   │       ├── data_generators.py      # Synthetic data generation
│   │       ├── missingness_patterns.py  # Missingness pattern classes
│   │       ├── imputation_methods.py   # Imputation method classes
│   │       ├── evaluator.py            # Evaluation metrics
│   │       └── simulator.py            # Simulation orchestration
│   └── analysis/
│       └── compare_imputation_methods.py  # Results analysis & visualization
├── tests/
│   ├── test_simulation_pipeline.py     # Main test suite
│   ├── test_regression.py              # Regression tests
│   └── test_compare_imputation_methods.py
├── scripts/
│   ├── benchmark.py                   # Benchmarking script
│   ├── complexity_analysis.py        # Complexity analysis
│   ├── performance_comparison.py      # Performance comparison
│   └── stability_check.py             # Numerical stability checks
├── docs/
│   ├── BASELINE.md                    # Baseline performance documentation
│   ├── OPTIMIZATION.md                # Optimization documentation
│   ├── PERFORMANCE_VISUALIZATIONS.md  # Performance visualizations
│   ├── REGRESSION_TESTING.md          # Regression testing documentation
│   └── simulation_workflow.tex         # LaTeX workflow documentation
└── results/
    ├── report/                        # Simulation results
    ├── figures/                       # Generated visualizations
    └── tables/                        # Summary tables
```

## Output Files

### Simulation Results

- `results/report/*/results_all_runs.csv`: All individual run results
- `results/report/*/results_averaged.csv`: Aggregated statistics across runs

### Analysis Outputs

- `results/figures/*.png`: Visualization figures
- `results/tables/combined_results_averaged.csv`: Meta-averaged results
- `results/tables/statistical_tests.csv`: Statistical test results

## Testing

To ensure the integrity and stability of the simulation logic:

1. **Navigate to the project root**
2. **Execute Pytest**: `make test` or `pytest tests/ -v`

The test suite verifies:
- Core simulation mechanisms
- Metric calculations (Log Loss, MSE, R²)
- Simulation uncertainty (STD across runs)
- Regression tests (optimization correctness)
- Edge cases and numerical stability

## Configuration

### JSON Configuration Format

```json
{
    "n": [50, 100],                    // Sample sizes to test
    "p": [5, 10],                      // Number of predictors
    "num_runs": 2,                     // Monte Carlo replicates per combination
    "continuous_pct": [0.4],          // Proportion of continuous covariates
    "integer_pct": [0.4],             // Proportion of integer covariates
    "sparsity": [0.3],                // Missingness sparsity level
    "include_interactions": [false],   // Include interaction terms
    "include_nonlinear": [false],      // Include nonlinear terms
    "include_splines": [false],        // Include spline terms
    "seed": 123                        // Random seed for reproducibility
}
```

## Documentation

- **`docs/BASELINE.md`**: Baseline performance characteristics
- **`docs/OPTIMIZATION.md`**: Implemented optimizations and their impact
- **`docs/PERFORMANCE_VISUALIZATIONS.md`**: Performance comparison visualizations
- **`docs/REGRESSION_TESTING.md`**: Regression testing methodology and results
- **`docs/simulation_workflow.tex`**: LaTeX documentation of workflow structure

## Contributing

Feel free to contribute by:
- Adding new imputation methods
- Implementing new missingness patterns
- Improving simulation scenarios
- Enhancing analysis scripts
- Improving documentation

Please follow the project structure and update project documentation as needed.

## License

This project is intended for research purposes in the context of Duke-NUS Summer Research.
