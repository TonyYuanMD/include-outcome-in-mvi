--- Configuration ---

PYTHON = python
PYTEST = pytest

The base directory where all simulation outputs are stored

RESULTS_DIR = results

The directory containing the analysis script

ANALYSIS_SCRIPT = src/analysis/compare_imputation_methods.py

The script that runs the simulation

SIMULATION_SCRIPT = run_simulation.py

.PHONY: all simulate analyze figures clean test

--- Main Targets ---

all: simulate analyze figures
@echo "========================================================="
@echo "COMPLETE PIPELINE: Simulation, Analysis, and Visualization finished."
@echo "Results are available in the ${RESULTS_DIR}/ directory."
@echo "========================================================="

simulate:
@echo "--- 1. RUNNING SIMULATION AND GENERATING RAW RESULTS ---"
$(PYTHON) $(SIMULATION_SCRIPT)
@echo "Simulation complete. Raw results saved to ${RESULTS_DIR}/report/..."

analyze:
@echo "--- 2. PROCESSING RAW RESULTS AND GENERATING SUMMARY TABLES ---"
# Note: The compare_imputation_methods.py script generates tables AND figures.
$(PYTHON) $(ANALYSIS_SCRIPT)
@echo "Analysis complete. Summary tables saved to ${RESULTS_DIR}/tables/."

figures: analyze
@echo "--- 3. CREATING VISUALIZATIONS ---"
# Since the analyze step runs the visualization logic, we just re-run the script
# to ensure figures are generated if they weren't in the first 'analyze' step.
$(PYTHON) $(ANALYSIS_SCRIPT)
@echo "Visualizations complete. Figures saved to ${RESULTS_DIR}/figures/."

test:
@echo "--- 4. RUNNING TEST SUITE (Pytest) ---"
# Runs pytest from the project root against the tests/ directory
$(PYTEST) tests/
@echo "Test suite finished."

clean:
@echo "--- 5. CLEANING GENERATED FILES ---"
# Remove the main results directory
-rm -rf $(RESULTS_DIR)
# Remove any generated log files
-rm -f simulation.log.txt
@echo "Clean complete. Removed $(RESULTS_DIR)/ and log files."