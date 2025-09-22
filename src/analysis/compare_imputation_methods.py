import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        return None
    results_all = pd.read_csv(results_all_path)
    results_avg = pd.read_csv(results_avg_path) if os.path.exists(results_avg_path) else None
    return results_all, results_avg

def perform_statistical_tests(combined_results):
    """Run ANOVA to test if methods differ significantly by metric."""
    tests = {}
    for metric in ['rmse', 'mae']:  # Adjust to your metrics
        if metric in combined_results.columns:
            f_stat, p_value = stats.f_oneway(
                *[combined_results[combined_results['method'] == m][metric].dropna() for m in combined_results['method'].unique()]
            )
            tests[metric] = {'f_stat': f_stat, 'p_value': p_value}
            logger.info(f"ANOVA for {metric}: F={f_stat:.2f}, p={p_value:.3f}")
    return tests

def compare_methods(report_dirs):
    """Main function to compare imputation methods across parameter sets."""
    all_results = []
    for report_dir in report_dirs:
        results_all, results_avg = load_results(report_dir)
        if results_all is not None:
            results_all['param_set'] = os.path.basename(report_dir)
            all_results.append(results_all)
    
    if not all_results:
        logger.error("No valid results found.")
        return
    
    combined_results = pd.concat(all_results, ignore_index=True)
    logger.info(f"Combined results shape: {combined_results.shape}")
    logger.info(f"Metrics available: {combined_results.select_dtypes(include='number').columns.tolist()}")
    
    # Statistical tests
    tests = perform_statistical_tests(combined_results)
    
    # Save combined results
    tables_dir = 'results/tables/'
    os.makedirs(tables_dir, exist_ok=True)
    combined_results.to_csv(os.path.join(tables_dir, 'combined_results_all.csv'), index=False)
    pd.DataFrame(tests).to_csv(os.path.join(tables_dir, 'statistical_tests.csv'))
    
    # Pivot table for comparison (e.g., mean RMSE by method and missingness, averaged over param_sets)
    for metric in ['rmse', 'mae']:  # Adjust as needed
        if metric in combined_results.columns:
            pivot_data = combined_results.pivot_table(
                values=metric, index='method', columns='missingness', aggfunc='mean'
            )
            pivot_data.to_csv(os.path.join(tables_dir, f'{metric}_pivot_by_method_missingness.csv'))
            logger.info(f"Saved pivot for {metric}")
    
    # Visualizations
    figures_dir = 'results/figures/'
    os.makedirs(figures_dir, exist_ok=True)
    
    
    
    # 2. Heatmap: Mean RMSE by method and missingness (averaged over param_sets)
    avg_rmse = combined_results.pivot_table(values='rmse', index='method', columns='missingness', aggfunc='mean')
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_rmse, annot=True, cmap='YlOrRd', fmt='.3f')
    plt.title('Mean RMSE Heatmap: Methods vs. Missingness Types')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'rmse_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Bar plot: Mean RMSE by param_set (top methods only for clarity)
    # top_methods = combined_results.groupby('method')['rmse'].mean().nlargest(5).index
    # subset = combined_results[combined_results['method'].isin(top_methods)]
    # plt.figure(figsize=(12, 6))
    # sns.barplot(data=subset, x='param_set', y='rmse', hue='method')
    # plt.title('Mean RMSE for Top 5 Methods by Sample Size (n)')
    # plt.tight_layout()
    # plt.savefig(os.path.join(figures_dir, 'rmse_bar_by_sample_size.png'), dpi=300, bbox_inches='tight')
    # plt.close()
    g = sns.catplot(
        data=combined_results, 
        x='method', 
        y='rmse', 
        hue='missingness', 
        col='param_set', 
        kind='box', 
        height=6, 
        aspect=1.2,
        col_wrap=2  # Limits to 2 columns of facets
    )
    g.set_axis_labels('Imputation Method', 'RMSE')
    g.set_titles(col_template="{col_name}")
    plt.suptitle('RMSE by Imputation Method and Missingness (Faceted by Sample Size)', y=1.05)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'rmse_boxplot_by_method.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Analysis complete. Tables in {tables_dir}, figures in {figures_dir}")

if __name__ == "__main__":
    report_dirs = discover_report_dirs()
    compare_methods(report_dirs)