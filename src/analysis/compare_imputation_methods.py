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

def discover_report_dirs(base_dir='results/report/'):
    """Dynamically find all report directories."""
    report_dirs = list(Path(base_dir).glob('n_*'))
    if not report_dirs:
        logger.error(f"No report directories found in {base_dir}")
        return []
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
    
    # 1. Log Loss Test (Binary Outcome)
    metric = 'y_log_loss_mean'
    if metric in combined_results.columns:
        # We need the mean metric aggregated by Run, Method, and Missingness
        grouped = combined_results.groupby(['run_idx', 'method', 'missingness'])[metric].mean().reset_index()
        
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
        grouped = combined_results.groupby(['run_idx', 'method', 'missingness'])[metric].mean().reset_index()
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
            all_results_avg.append(results_avg)
        else:
            logger.warning(f"Skipping {report_dir} due to missing or invalid averaged data")
    
    if not all_results_avg:
        logger.error("No valid averaged results found.")
        return
    
    # Use averaged results for visualization and statistical tests
    combined_results_avg = pd.concat(all_results_avg, ignore_index=True)
    logger.info(f"Combined averaged results shape: {combined_results_avg.shape}")
    
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
    # tests = perform_statistical_tests(combined_results_avg) 
    
    # Save combined results (Use the averaged dataframe for simpler output)
    tables_dir = 'results/tables/'
    os.makedirs(tables_dir, exist_ok=True)
    df.to_csv(os.path.join(tables_dir, 'combined_results_averaged.csv'), index=False)
    # pd.DataFrame(tests).to_csv(os.path.join(tables_dir, 'statistical_tests.csv'))
    
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

    plt.axhline(y=0.693, color='r', linestyle='--', linewidth=1, label='Random Guess (0.693)')
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
    plt.axvline(x=0.693, color='r', linestyle='--', linewidth=1, label='Random Baseline')
    
    plt.legend(title='Method/Missingness', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'y_log_loss_stability_plot.png'), dpi=300)
    plt.close()

    logger.info(f"Analysis complete. Tables in {tables_dir}, figures in {figures_dir}")

if __name__ == "__main__":
    # 1. Define the specific directory path you want to analyze.
    # NOTE: You must replace the placeholder below with the actual folder name
    # generated by your run (it will be inside the 'results/report/' folder).
    TARGET_DIR_SUFFIX = 'n_100_100_p_10_10_runs_10_cont_0.4_0.4_int_0.4_0.4_sparse_0.3_0.3_inter_0_0_nonlin_0_0_splines_0_0'
    
    # 2. Construct the full path and ensure it's in a list format for compare_methods
    BASE_DIR = 'results/report/'
    target_dir_full = os.path.join(BASE_DIR, TARGET_DIR_SUFFIX)
    
    if os.path.isdir(target_dir_full):
        report_dirs = [target_dir_full]
        compare_methods(report_dirs)
    else:
        logger.error(f"Analysis aborted: Target directory not found at {target_dir_full}")