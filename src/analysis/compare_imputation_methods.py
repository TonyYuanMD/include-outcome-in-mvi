import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions (Unchanged) ---

def discover_report_dirs(base_dir='results/report/', use_latest_only=False):
    """
    Dynamically find all report directories.
    
    Parameters:
    -----------
    base_dir : str
        Base directory to search for report directories
    use_latest_only : bool, default=False
        If True, return only the most recently modified directory.
        If False, return all directories matching the pattern.
    
    Returns:
    --------
    list : List of report directory paths
    """
    # Look for directories matching various patterns (n_*, cpu_n_*, gpu_n_*, etc.)
    base_path = Path(base_dir)
    if not base_path.exists():
        logger.error(f"Base directory does not exist: {base_dir}")
        return []
    
    # Get all subdirectories
    all_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    # Filter for directories that look like report directories (contain results_averaged.csv)
    # and match common naming patterns
    report_dirs = []
    for d in all_dirs:
        results_file = d / 'results_averaged.csv'
        if results_file.exists() and results_file.stat().st_size > 0:
            # Check if directory name suggests it's a report directory
            # (contains 'n_' or 'cpu_n_' or 'gpu_n_' etc.)
            if 'n_' in d.name or 'cpu' in d.name.lower() or 'gpu' in d.name.lower():
                report_dirs.append(d)
    
    if not report_dirs:
        logger.error(f"No valid report directories found in {base_dir} (all are empty or missing results_averaged.csv)")
        return []
    
    if use_latest_only:
        # Sort by modification time of results_averaged.csv, get the most recent
        report_dirs = sorted(report_dirs, key=lambda p: (p / 'results_averaged.csv').stat().st_mtime, reverse=True)
        latest_dir = report_dirs[0]
        logger.info(f"Using only the most recent directory: {latest_dir.name}")
        return [str(latest_dir)]
    else:
        logger.info(f"Found {len(report_dirs)} report directories: {[d.name for d in report_dirs]}")
        return [str(d) for d in report_dirs]

def load_results(report_dir):
    """Load results_all_runs.csv and results_averaged.csv from a report directory."""
    results_all_path = os.path.join(report_dir, 'results_all_runs.csv')
    results_avg_path = os.path.join(report_dir, 'results_averaged.csv')
    if not os.path.exists(results_all_path):
        logger.warning(f"Missing results_all_runs.csv in {report_dir}")
        return None, None
    results_all = pd.read_csv(results_all_path)
    results_avg = pd.read_csv(results_avg_path) if os.path.exists(results_avg_path) else None
    
    # Check for core columns. If not present, log a warning (often due to old format)
    if not all(col in results_all.columns for col in ['n', 'p']):
        logger.warning(f"Results file in {report_dir} seems to be missing parsed parameter columns.")
        
    return results_all, results_avg

# --- Statistical Tests (Adjusted for Actual Metrics) ---

def perform_statistical_tests(combined_results):
    """Run ANOVA to test if methods differ significantly by metric (Log Loss and R2)."""
    tests = {}
    
    # Check if we have run_idx (from results_all) or if we're using averaged results
    has_run_idx = 'run_idx' in combined_results.columns
    
    # 1. Log Loss Test (Binary Outcome)
    metric = 'y_log_loss_mean'
    if metric in combined_results.columns:
        if has_run_idx:
            # Use run_idx for proper ANOVA with multiple observations per group
            grouped = combined_results.groupby(['run_idx', 'method', 'missingness'])[metric].mean().reset_index()
        else:
            # For averaged results, group by method and missingness only
            # Note: ANOVA requires multiple observations per group, so this is limited
            grouped = combined_results.groupby(['method', 'missingness'])[metric].mean().reset_index()
            logger.warning(f"Using averaged results for {metric} - ANOVA may be limited without run-level data")
        
        # Run ANOVA on the performance metric for each method
        methods = grouped['method'].unique()
        data_to_test = [grouped[grouped['method'] == m][metric].dropna() for m in methods]
        
        # Filter out methods with no data (empty arrays)
        data_to_test = [data for data in data_to_test if len(data) > 0]
        
        if len(data_to_test) >= 2:
            f_stat, p_value = stats.f_oneway(*data_to_test)
            tests[metric] = {'f_stat': f_stat, 'p_value': p_value}
            logger.info(f"ANOVA for {metric}: F={f_stat:.2f}, p={p_value:.3f}")
        else:
            logger.warning(f"Not enough groups to run ANOVA for {metric}.")
            
    # 2. R2 Test (Continuous Outcome)
    metric = 'y_score_r2_mean'
    if metric in combined_results.columns:
        if has_run_idx:
            grouped = combined_results.groupby(['run_idx', 'method', 'missingness'])[metric].mean().reset_index()
        else:
            grouped = combined_results.groupby(['method', 'missingness'])[metric].mean().reset_index()
            logger.warning(f"Using averaged results for {metric} - ANOVA may be limited without run-level data")
            
        methods = grouped['method'].unique()
        data_to_test = [grouped[grouped['method'] == m][metric].dropna() for m in methods]
        data_to_test = [data for data in data_to_test if len(data) > 0]

        if len(data_to_test) >= 2:
            f_stat, p_value = stats.f_oneway(*data_to_test)
            tests[metric] = {'f_stat': f_stat, 'p_value': p_value}
            logger.info(f"ANOVA for {metric}: F={f_stat:.2f}, p={p_value:.3f}")
        else:
            logger.warning(f"Not enough groups to run ANOVA for {metric}.")
            
    return tests

# --- Main Comparison Function (Updated) ---

def compare_methods(report_dirs):
    """Main function to compare imputation methods across parameter sets."""
    all_results_avg = []
    
    for report_dir in report_dirs:
        results_all, results_avg = load_results(report_dir)
        if results_avg is not None:
            # Add source identifier (CPU vs GPU) based on directory name
            dir_name = os.path.basename(os.path.normpath(report_dir))
            if dir_name.startswith('cpu'):
                results_avg['source'] = 'CPU'
            elif dir_name.startswith('gpu'):
                results_avg['source'] = 'GPU'
            else:
                results_avg['source'] = 'Unknown'
            all_results_avg.append(results_avg)
        else:
            logger.warning(f"Skipping {report_dir} due to missing or invalid averaged data")
    
    if not all_results_avg:
        logger.error("No valid averaged results found.")
        return
    
    # Use averaged results for visualization and statistical tests
    combined_results_avg = pd.concat(all_results_avg, ignore_index=True)
    logger.info(f"Combined averaged results shape: {combined_results_avg.shape}")
    logger.info(f"Source distribution: {combined_results_avg['source'].value_counts().to_dict()}")
    
    # Rename the new run-level STD columns for better plotting names
    # Example: 'y_log_loss_mean_std_runs' -> 'Log_Loss_STD_Runs'
    rename_map = {
        'y_log_loss_mean_std_runs': 'Log_Loss_STD_Runs',
        'y_score_r2_mean': 'R2_Mean',
        'y_score_r2_mean_std_runs': 'R2_STD_Runs',
        'y_log_loss_mean': 'Log_Loss_Mean'
    }
    df = combined_results_avg.rename(columns=rename_map)

    # Statistical tests (requires results_all for run_idx)
    # NOTE: Since we only loaded results_avg, this section might be limited.
    # For full ANOVA, run_single_combination should return both results_all and results_avg.
    # We will skip the complex ANOVA for this integration, as it requires the un-aggregated run data.
    tests = perform_statistical_tests(combined_results_avg) 
    
    # Save combined results (Use the averaged dataframe for simpler output)
    tables_dir = 'results/tables/'
    os.makedirs(tables_dir, exist_ok=True)
    df.to_csv(os.path.join(tables_dir, 'combined_results_averaged.csv'), index=False)
    pd.DataFrame(tests).to_csv(os.path.join(tables_dir, 'statistical_tests.csv'))
    
    # Setup plotting environment
    figures_dir = 'results/figures/'
    os.makedirs(figures_dir, exist_ok=True)
    
    # --- VISUALIZATION 1: Heatmap (R2 Mean) ---
    logger.info("Generating R2 Heatmap...")
    df_r2_pivot = df.groupby(['method', 'missingness'])['R2_Mean'].mean().unstack()
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        df_r2_pivot, annot=True, fmt=".3f", cmap='viridis', linewidths=.5,
        cbar_kws={'label': 'Mean $R^2$ for $Y_{score}$'}
    )
    plt.title('Mean $R^2$ Performance of Imputation Methods by Missingness Pattern')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'y_score_r2_heatmap_methods_vs_missingness.png'), dpi=300)
    plt.close()
    
    # --- VISUALIZATION 2: Grouped Bar Plot (Log Loss & Outcome Inclusion) ---
    logger.info("Generating Log Loss Bar Plot...")
    
    # Prepare data for bar plot (focus on MAR for example)
    df_plot_mar = df[
        (df['missingness'] == 'mar') &
        (~df['method'].isin(['complete_data', 'mean'])) 
    ].copy()

    order_methods = df_plot_mar.groupby('method')['Log_Loss_Mean'].mean().sort_values().index

    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=df_plot_mar, x='method', y='Log_Loss_Mean', hue='imputation_outcome_used',
        order=order_methods, capsize=0.05, palette='Set2'
    )
    
    # Add error bars for Simulation Uncertainty (STD of runs)
    means = df_plot_mar.groupby(['method', 'imputation_outcome_used'])['Log_Loss_Mean'].mean()
    errors = df_plot_mar.groupby(['method', 'imputation_outcome_used'])['Log_Loss_STD_Runs'].mean()
    
    # Use unique pairs of (method, outcome) to get bar positions and heights
    unique_combinations = df_plot_mar[['method', 'imputation_outcome_used']].drop_duplicates()
    
    # Recalculate positions based on the current bar plot structure for correct overlay
    x_pos = np.arange(len(order_methods))
    bar_width = 0.8 / df_plot_mar['imputation_outcome_used'].nunique()
    
    for i, outcome_use in enumerate(df_plot_mar['imputation_outcome_used'].unique()):
        subset = df_plot_mar[df_plot_mar['imputation_outcome_used'] == outcome_use]
        x_centers = [x_pos[order_methods.get_loc(m)] + (i - 1) * bar_width for m in subset['method'].unique()]
        
        # Ensure we have means and errors for the current subset
        subset_means = means.loc[(subset['method'], outcome_use)].values
        subset_errors = errors.loc[(subset['method'], outcome_use)].values

        plt.errorbar(
            x=x_centers,
            y=subset_means,
            yerr=subset_errors,
            fmt='none',
            c='black',
            capsize=4
        )

    # plt.axhline(y=0.693, color='r', linestyle='--', linewidth=1, label='Random Guess (0.693)')
    plt.title('Effect of Outcome Inclusion on Binary Prediction Utility (MAR Missingness)')
    plt.ylabel('Mean Log Loss for $Y$ (Lower is Better)')
    plt.xlabel('Imputation Method')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Outcome Used in Imputation', loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'y_log_loss_outcome_inclusion_mar_barplot.png'), dpi=300)
    plt.close()
    
    # --- VISUALIZATION 3: Stability Plot (Log Loss Mean vs. Log Loss STD) ---
    logger.info("Generating Log Loss Stability Plot...")

    # Filter for methods that don't include outcome to test core method stability
    df_stable = df[df['imputation_outcome_used'] == 'none'].copy() 

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_stable,
        x='Log_Loss_Mean',
        y='Log_Loss_STD_Runs',
        hue='method',
        style='missingness',
        s=150, # Size of points
        alpha=0.8
    )

    plt.title('Log Loss: Performance (Mean) vs. Stability (STD Across Runs)')
    plt.xlabel('Mean Log Loss for Y (Lower is Better)')
    plt.ylabel('STD of Mean Log Loss Across Runs (Lower is More Stable)')
    
    # Add a visual guide for the ideal region
    # plt.axvline(x=0.693, color='r', linestyle='--', linewidth=1, label='Random Baseline')
    
    plt.legend(title='Method/Missingness', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'y_log_loss_stability_plot.png'), dpi=300)
    plt.close()
    
    # --- VISUALIZATION 4: CPU vs GPU Comparison (if source column exists) ---
    if 'source' in df.columns and df['source'].nunique() > 1:
        logger.info("Generating CPU vs GPU Comparison Plot...")
        
        # Filter for methods that exist in both CPU and GPU
        methods_in_both = []
        for method in df['method'].unique():
            method_df = df[df['method'] == method]
            if method_df['source'].nunique() > 1:
                methods_in_both.append(method)
        
        if methods_in_both:
            # Create comparison plot for Log Loss
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            
            # Plot 1: Log Loss comparison
            ax1 = axes[0]
            comparison_data = df[df['method'].isin(methods_in_both)].copy()
            comparison_pivot = comparison_data.pivot_table(
                values='Log_Loss_Mean', 
                index='method', 
                columns='source', 
                aggfunc='mean'
            )
            
            x = np.arange(len(comparison_pivot.index))
            width = 0.35
            
            if 'CPU' in comparison_pivot.columns and 'GPU' in comparison_pivot.columns:
                ax1.bar(x - width/2, comparison_pivot['CPU'], width, label='CPU', alpha=0.8)
                ax1.bar(x + width/2, comparison_pivot['GPU'], width, label='GPU', alpha=0.8)
                ax1.set_xlabel('Imputation Method')
                ax1.set_ylabel('Mean Log Loss (Lower is Better)')
                ax1.set_title('CPU vs GPU: Log Loss Performance')
                ax1.set_xticks(x)
                ax1.set_xticklabels(comparison_pivot.index, rotation=45, ha='right')
                ax1.legend()
                ax1.grid(True, linestyle='--', alpha=0.3)
            
            # Plot 2: R² comparison
            ax2 = axes[1]
            comparison_pivot_r2 = comparison_data.pivot_table(
                values='R2_Mean', 
                index='method', 
                columns='source', 
                aggfunc='mean'
            )
            
            if 'CPU' in comparison_pivot_r2.columns and 'GPU' in comparison_pivot_r2.columns:
                ax2.bar(x - width/2, comparison_pivot_r2['CPU'], width, label='CPU', alpha=0.8)
                ax2.bar(x + width/2, comparison_pivot_r2['GPU'], width, label='GPU', alpha=0.8)
                ax2.set_xlabel('Imputation Method')
                ax2.set_ylabel('Mean R² (Higher is Better)')
                ax2.set_title('CPU vs GPU: R² Performance')
                ax2.set_xticks(x)
                ax2.set_xticklabels(comparison_pivot_r2.index, rotation=45, ha='right')
                ax2.legend()
                ax2.grid(True, linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, 'cpu_vs_gpu_comparison.png'), dpi=300)
            plt.close()
            
            logger.info(f"CPU vs GPU comparison includes {len(methods_in_both)} methods: {methods_in_both}")

    logger.info(f"Analysis complete. Tables in {tables_dir}, figures in {figures_dir}")

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare imputation methods across simulation results')
    parser.add_argument('--latest', '-l', action='store_true', 
                       help='Analyze only the most recent report directory')
    parser.add_argument('--dir', '-d', type=str, default=None, nargs='+',
                       help='Analyze specific report directory(ies) (relative to results/report/ or absolute path). Can specify multiple directories.')
    parser.add_argument('--base-dir', type=str, default='results/report/',
                       help='Base directory to search for report directories (default: results/report/)')
    
    args = parser.parse_args()
    
    BASE_DIR = args.base_dir
    
    # If specific directory(ies) is provided, use it/them
    if args.dir:
        report_dirs = []
        for dir_arg in args.dir:
            # Check if it's an absolute path or relative
            if os.path.isabs(dir_arg):
                report_dir = dir_arg
            else:
                # Try relative to base_dir first, then relative to current directory
                if os.path.exists(os.path.join(BASE_DIR, dir_arg)):
                    report_dir = os.path.join(BASE_DIR, dir_arg)
                elif os.path.exists(dir_arg):
                    report_dir = dir_arg
                else:
                    logger.error(f"Directory not found: {dir_arg}")
                    sys.exit(1)
            
            # Verify it has results_averaged.csv
            if not os.path.exists(os.path.join(report_dir, 'results_averaged.csv')):
                logger.error(f"Directory {report_dir} does not contain results_averaged.csv")
                sys.exit(1)
            
            report_dirs.append(report_dir)
        
        logger.info(f"Analyzing {len(report_dirs)} specific directory(ies): {[os.path.basename(d) for d in report_dirs]}")
    else:
        # Automatically discover all report directories
        use_latest_only = args.latest
        report_dirs = discover_report_dirs(BASE_DIR, use_latest_only=use_latest_only)
    
    if report_dirs:
        logger.info(f"Found {len(report_dirs)} report directory(ies) to analyze")
        compare_methods(report_dirs)
    else:
        logger.error(f"Analysis aborted: No report directories found")
        logger.error("Please run 'make simulate' first to generate simulation results.")