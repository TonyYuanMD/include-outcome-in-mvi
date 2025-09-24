import os
import pytest
from include_y_python.src.analysis.compare_imputation_methods import discover_report_dirs, load_results, compare_methods
import logging
from unittest.mock import patch, mock_open
import pandas as pd

# Configure logging for testing
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fixture to create a mock file system
@pytest.fixture
def mock_filesystem():
    with patch('os.path.exists') as mock_exists, patch('builtins.open', new_callable=mock_open) as mock_file:
        # Mock file existence
        mock_exists.return_value = False  # Default to non-existent files
        yield mock_exists, mock_file

# Fixture to set up a mock report directory structure
@pytest.fixture
def mock_report_dirs(tmp_path):
    report_dir = tmp_path / "results" / "report"
    report_dir.mkdir(parents=True)
    (report_dir / "n_50_p_5_runs_2_cont_0.4_sparse_0.3").mkdir()
    (report_dir / "n_50_p_5_runs_10_cont_0.4_sparse_0.3").mkdir()
    with patch('pathlib.Path.glob') as mock_glob:
        mock_glob.return_value = [report_dir / "n_50_p_5_runs_2_cont_0.4_sparse_0.3", report_dir / "n_50_p_5_runs_10_cont_0.4_sparse_0.3"]
        yield str(report_dir)

# Test warning for missing results_all_runs.csv
def test_missing_results_all_runs_csv(mock_filesystem, mock_report_dirs, caplog):
    mock_exists, _ = mock_filesystem
    # Simulate missing results_all_runs.csv for the second directory
    mock_exists.side_effect = lambda path: "n_50_p_5_runs_2_cont_0.4_sparse_0.3" in str(path) and "results_all_runs.csv" in str(path)

    with caplog.at_level(logging.WARNING):
        report_dirs = discover_report_dirs(mock_report_dirs)
        for report_dir in report_dirs:
            results_all, results_avg = load_results(report_dir)
    
    # Check that a warning was logged for the missing file
    assert any("Missing results_all_runs.csv in" in record.message for record in caplog.records)
    assert len([record for record in caplog.records if "Missing results_all_runs.csv" in record.message]) == 1

# Test error when no valid results are found
def test_no_valid_results(mock_filesystem, mock_report_dirs, caplog):
    mock_exists, _ = mock_filesystem
    # Simulate no existing results_all_runs.csv for any directory
    mock_exists.return_value = False

    with caplog.at_level(logging.ERROR):
        report_dirs = discover_report_dirs(mock_report_dirs)
        compare_methods(report_dirs)
    
    # Check that an error was logged when no valid results are found
    assert any("No valid results found." in record.message for record in caplog.records)
    assert len([record for record in caplog.records if "No valid results found." in record.message]) == 1

# Test successful load with valid file
def test_valid_results_load(mock_filesystem, mock_report_dirs, caplog):
    mock_exists, mock_file = mock_filesystem
    # Simulate existing results_all_runs.csv for the first directory
    mock_exists.side_effect = lambda path: "n_50_p_5_runs_2_cont_0.4_sparse_0.3" in str(path) and "results_all_runs.csv" in str(path)
    mock_file.return_value = pd.DataFrame({'method': ['mean'], 'missingness': ['mcar'], 'rmse': [1.0], 'n': [50], 'p': [5], 'runs': [2], 'cont_pct': [0.4], 'sparsity': [0.3], 'run': [0]}).to_csv(index=False)

    with caplog.at_level(logging.INFO):
        report_dirs = discover_report_dirs(mock_report_dirs)
        results_all, results_avg = load_results(report_dirs[0])
    
    # Check that no warnings or errors occurred and data was loaded
    assert not any(record.levelno >= logging.WARNING for record in caplog.records)
    assert results_all is not None
    assert results_all['n'].iloc[0] == 50
    assert results_all['p'].iloc[0] == 5