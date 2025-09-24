import os
import sys
sys.path.insert(0, os.path.dirname(__file__))  # Add the project root (include_y_python) to path

import pytest
from run_simulation import run_simulation  # Adjusted import to reflect root-level location
import logging

# Configure logging for testing
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test ValueError for invalid continuous_pct + sparsity (sum > 1)
def test_invalid_continuous_pct():
    with pytest.raises(ValueError) as exc_info:
        run_simulation(n=50, p=5, num_runs=1, continuous_pct=0.8, sparsity=0.3)  # sum = 1.1 > 1
    assert "integer_pct" in str(exc_info.value)
    assert "negative" in str(exc_info.value)

# Test small n (e.g., n=5), assuming it runs without error (no validation in code)
def test_small_n(caplog):
    with caplog.at_level(logging.INFO):
        run_simulation(n=5, p=5, num_runs=1, continuous_pct=0.4, sparsity=0.3)
    # If validation for small n is added in the future, check for warning
    # assert any("n too small" in record.message for record in caplog.records)
    assert "Simulation complete" in caplog.text  # Check that it completes