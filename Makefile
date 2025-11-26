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

.PHONY: all simulate analyze figures clean test help

# ===================================================================
# Main Targets
# ===================================================================

help:
	@echo "Available targets:"
	@echo "  make all          - Run complete pipeline: simulate -> analyze -> figures"
	@echo "  make simulate     - Run simulation study (use CONFIG=file.json for JSON config)"
	@echo "  make analyze      - Process results and generate summary tables"
	@echo "  make figures      - Generate visualization figures"
	@echo "  make test         - Run test suite"
	@echo "  make clean        - Remove generated files and results"
	@echo ""
	@echo "Examples:"
	@echo "  make simulate                    # Use default parameters"
	@echo "  make simulate CONFIG=config.json  # Use JSON configuration file"

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
else
	-rm -rf $(RESULTS_DIR)
	-rm -f simulation.log.txt
endif
	@echo "Clean complete. Removed $(RESULTS_DIR)/ and log files."