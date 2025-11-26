# ===================================================================
# Configuration
# ===================================================================

PYTHON = python
PYTEST = pytest

# Base directory where all simulation outputs are stored
RESULTS_DIR = results

# Directory containing the analysis script
ANALYSIS_SCRIPT = src/analysis/compare_imputation_methods.py

# Script that runs the simulation
SIMULATION_SCRIPT = run_simulation.py

# Optional JSON configuration file (can be overridden: make simulate CONFIG=my_config.json)
CONFIG ?= 

# Test directory
TEST_DIR = tests

.PHONY: all simulate analyze figures clean test help profile complexity benchmark parallel stability-check performance-comparison

# ===================================================================
# Main Targets
# ===================================================================

help:
	@echo "Available targets:"
	@echo "  make all              - Run complete pipeline: simulate -> analyze -> figures"
	@echo "  make simulate         - Run simulation study (use CONFIG=file.json for JSON config)"
	@echo "  make analyze          - Process results and generate summary tables"
	@echo "  make figures          - Generate visualization figures"
	@echo "  make test             - Run test suite"
	@echo "  make clean            - Remove generated files and results"
	@echo ""
	@echo "Performance & Analysis:"
	@echo "  make profile          - Run profiling on representative simulation"
	@echo "  make complexity       - Run computational complexity analysis (timing vs n, p)"
	@echo "  make benchmark            - Run timing comparison benchmarks"
	@echo "  make parallel             - Run optimized version with parallelization (default)"
	@echo "  make stability-check      - Check for warnings/convergence issues across conditions"
	@echo "  make performance-comparison - Generate baseline vs optimized visualizations"
	@echo ""
	@echo "Examples:"
	@echo "  make simulate                    # Use default parameters"
	@echo "  make simulate CONFIG=config.json  # Use JSON configuration file"
	@echo "  make profile                     # Profile performance"
	@echo "  make complexity                  # Analyze computational complexity"

all: simulate analyze figures
	@echo "========================================================="
	@echo "COMPLETE PIPELINE: Simulation, Analysis, and Visualization finished."
	@echo "Results are available in the $(RESULTS_DIR)/ directory."
	@echo "========================================================="

simulate:
	@echo "--- 1. RUNNING SIMULATION AND GENERATING RAW RESULTS ---"
ifeq ($(CONFIG),)
	@echo "Running simulation with default parameters..."
	$(PYTHON) $(SIMULATION_SCRIPT)
else
	@echo "Running simulation with configuration file: $(CONFIG)"
	$(PYTHON) -c "from run_simulation import run_simulation; run_simulation(config_file='$(CONFIG)')"
endif
	@echo "Simulation complete. Raw results saved to $(RESULTS_DIR)/report/..."

analyze:
	@echo "--- 2. PROCESSING RAW RESULTS AND GENERATING SUMMARY TABLES ---"
	@echo "Note: The compare_imputation_methods.py script generates tables AND figures."
	$(PYTHON) $(ANALYSIS_SCRIPT)
	@echo "Analysis complete. Summary tables saved to $(RESULTS_DIR)/tables/."

figures: analyze
	@echo "--- 3. CREATING VISUALIZATIONS ---"
	@echo "Since the analyze step runs the visualization logic, we just re-run the script"
	@echo "to ensure figures are generated if they weren't in the first 'analyze' step."
	$(PYTHON) $(ANALYSIS_SCRIPT)
	@echo "Visualizations complete. Figures saved to $(RESULTS_DIR)/figures/."

test:
	@echo "--- 4. RUNNING TEST SUITE (Pytest) ---"
	@echo "Runs pytest from the project root against the $(TEST_DIR)/ directory"
	$(PYTEST) $(TEST_DIR)/ -v
	@echo "Test suite finished."

clean:
	@echo "--- 5. CLEANING GENERATED FILES ---"
	@echo "Removing the main results directory..."
ifeq ($(OS),Windows_NT)
	-if exist $(RESULTS_DIR) rmdir /s /q $(RESULTS_DIR)
	-if exist simulation.log.txt del /f simulation.log.txt
	-if exist profiling_warnings.log del /f profiling_warnings.log
	-if exist stability_check.log del /f stability_check.log
else
	-rm -rf $(RESULTS_DIR)
	-rm -f simulation.log.txt
	-rm -f profiling_warnings.log
	-rm -f stability_check.log
endif
	@echo "Clean complete. Removed $(RESULTS_DIR)/ and log files."

# ===================================================================
# Performance & Analysis Targets
# ===================================================================

profile:
	@echo "--- RUNNING PERFORMANCE PROFILING ---"
	@echo "This will run profiling on representative simulations."
	@echo "Results will be saved to docs/ directory."
	$(PYTHON) profile_simulation.py
	@echo "Profiling complete. Check docs/ for results."

complexity:
	@echo "--- RUNNING COMPUTATIONAL COMPLEXITY ANALYSIS ---"
	@echo "This will test runtime vs n and p to analyze complexity."
	@echo "Results will be saved to docs/complexity_*.csv and *.png"
	$(PYTHON) scripts/complexity_analysis.py
	@echo "Complexity analysis complete. Check docs/ for results."

benchmark:
	@echo "--- RUNNING BENCHMARK TESTS ---"
	@echo "This will run timing benchmarks across different parameter sets."
	@echo "Results will be saved to docs/benchmark_results.csv"
	$(PYTHON) scripts/benchmark.py
	@echo "Benchmark complete. Check docs/benchmark_results.csv for results."

parallel:
	@echo "--- RUNNING OPTIMIZED SIMULATION WITH PARALLELIZATION ---"
	@echo "Note: Parallelization is already enabled by default in run_simulation.py"
	@echo "This target runs a simulation to demonstrate parallel execution."
ifeq ($(CONFIG),)
	@echo "Running simulation with default parameters (4 parallel processes)..."
	$(PYTHON) $(SIMULATION_SCRIPT)
else
	@echo "Running simulation with configuration file: $(CONFIG) (4 parallel processes)..."
	$(PYTHON) -c "from run_simulation import run_simulation; run_simulation(config_file='$(CONFIG)')"
endif
	@echo "Parallel simulation complete. Results saved to $(RESULTS_DIR)/report/..."

stability-check:
	@echo "--- RUNNING NUMERICAL STABILITY CHECK ---"
	@echo "This will test various parameter combinations and capture warnings/errors."
	@echo "Results will be saved to docs/stability_*.csv"
	$(PYTHON) scripts/stability_check.py
	@echo "Stability check complete. Check docs/stability_*.csv for results."

performance-comparison:
	@echo "--- GENERATING PERFORMANCE COMPARISON VISUALIZATIONS ---"
	@echo "This will create visualizations comparing baseline vs optimized performance."
	@echo "Results will be saved to docs/performance_*.png and performance_comparison.csv"
	$(PYTHON) scripts/performance_comparison.py
	@echo "Performance comparison complete. Check docs/performance_*.png for visualizations."