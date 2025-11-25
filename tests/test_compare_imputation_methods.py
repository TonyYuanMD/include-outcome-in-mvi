import os
import sys
import pytest
import logging
from unittest.mock import patch, mock_open
import pandas as pd

# Add parent directory to path to import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.compare_imputation_methods import discover_report_dirs, load_results, compare_methods

# Configure logging for testing
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
def test_missing_results_all_runs_csv(mock_report_dirs, caplog):
    # Create real files for one directory, leave the other without files
    report_dir = mock_report_dirs
    dir1 = os.path.join(report_dir, "n_50_p_5_runs_2_cont_0.4_sparse_0.3")
    dir2 = os.path.join(report_dir, "n_50_p_5_runs_10_cont_0.4_sparse_0.3")
    
    # Create results file only for dir1
    os.makedirs(dir1, exist_ok=True)
    os.makedirs(dir2, exist_ok=True)
    test_df = pd.DataFrame({'method': ['mean'], 'missingness': ['mcar'], 'n': [50], 'p': [5]})
    test_df.to_csv(os.path.join(dir1, 'results_all_runs.csv'), index=False)
    # dir2 has no file - should trigger warning

    with caplog.at_level(logging.WARNING):
        report_dirs = discover_report_dirs(report_dir)
        for report_dir_path in report_dirs:
            results_all, results_avg = load_results(report_dir_path)
    
    # Check that a warning was logged for the missing file
    assert any("Missing results_all_runs.csv in" in record.message for record in caplog.records)

# Test error when no valid results are found
def test_no_valid_results(mock_report_dirs, caplog):
    # Don't create any result files - should trigger error
    report_dir = mock_report_dirs

    with caplog.at_level(logging.ERROR):
        report_dirs = discover_report_dirs(report_dir)
        compare_methods(report_dirs)
    
    # Check that an error was logged when no valid results are found
    assert any("No valid" in record.message for record in caplog.records)

# Test successful load with valid file
def test_valid_results_load(mock_report_dirs, caplog):
    # Create real result files
    report_dir = mock_report_dirs
    dir1 = os.path.join(report_dir, "n_50_p_5_runs_2_cont_0.4_sparse_0.3")
    os.makedirs(dir1, exist_ok=True)
    
    test_df = pd.DataFrame({
        'method': ['mean'], 
        'missingness': ['mcar'], 
        'n': [50], 
        'p': [5], 
        'runs': [2], 
        'cont_pct': [0.4], 
        'sparsity': [0.3], 
        'run_idx': [0]
    })
    test_df.to_csv(os.path.join(dir1, 'results_all_runs.csv'), index=False)
    test_df.to_csv(os.path.join(dir1, 'results_averaged.csv'), index=False)

    with caplog.at_level(logging.INFO):
        report_dirs = discover_report_dirs(report_dir)
        results_all, results_avg = load_results(report_dirs[0])
    
    # Check that no warnings or errors occurred and data was loaded
    assert not any(record.levelno >= logging.WARNING for record in caplog.records)
    assert results_all is not None
    assert len(results_all) > 0
    assert results_all['n'].iloc[0] == 50
    assert results_all['p'].iloc[0] == 5