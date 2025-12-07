"""
Aggregate and analyze results from multiple experiment runs.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import argparse
from scipy import stats


def aggregate_results(results_csv: str) -> pd.DataFrame:
    """Aggregate results grouped by experimental conditions.
    
    Args:
        results_csv: Path to results CSV file
        
    Returns:
        DataFrame with aggregated statistics
    """
    df = pd.read_csv(results_csv)
    
    # Group by experimental conditions
    group_cols = ['dataset', 'backbone', 'method', 'shots', 'rank', 'reg_weight']
    group_cols = [col for col in group_cols if col in df.columns]
    
    # Compute statistics
    agg_funcs = {
        'val_acc': ['mean', 'std', 'min', 'max', 'count'],
        'val_f1': ['mean', 'std', 'min', 'max']
    }
    
    aggregated = df.groupby(group_cols).agg(agg_funcs).reset_index()
    
    # Flatten column names
    aggregated.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                         for col in aggregated.columns.values]
    
    # Compute 95% confidence intervals
    def compute_ci(group):
        acc_values = group['val_acc'].values
        f1_values = group['val_f1'].values
        
        if len(acc_values) > 1:
            acc_ci = stats.t.interval(0.95, len(acc_values)-1,
                                     loc=np.mean(acc_values),
                                     scale=stats.sem(acc_values))
            f1_ci = stats.t.interval(0.95, len(f1_values)-1,
                                    loc=np.mean(f1_values),
                                    scale=stats.sem(f1_values))
        else:
            acc_ci = (acc_values[0], acc_values[0])
            f1_ci = (f1_values[0], f1_values[0])
            
        return pd.Series({
            'acc_ci_low': acc_ci[0],
            'acc_ci_high': acc_ci[1],
            'f1_ci_low': f1_ci[0],
            'f1_ci_high': f1_ci[1]
        })
    
    ci_df = df.groupby(group_cols).apply(compute_ci).reset_index()
    aggregated = aggregated.merge(ci_df, on=group_cols)
    
    return aggregated


def print_results_table(aggregated: pd.DataFrame, metric: str = 'val_acc') -> None:
    """Print formatted results table.
    
    Args:
        aggregated: Aggregated results DataFrame
        metric: Metric to display ('val_acc' or 'val_f1')
    """
    print(f"\n{'='*80}")
    print(f"Results Summary - {metric.upper()}")
    print(f"{'='*80}\n")
    
    for dataset in aggregated['dataset'].unique():
        dataset_df = aggregated[aggregated['dataset'] == dataset]
        print(f"\n{dataset.upper()}:")
        print("-" * 80)
        
        for _, row in dataset_df.iterrows():
            mean = row[f'{metric}_mean'] * 100
            std = row[f'{metric}_std'] * 100
            ci_low = row[f'{metric[4:]}_ci_low'] * 100  # val_acc -> acc
            ci_high = row[f'{metric[4:]}_ci_high'] * 100
            
            method_str = f"{row['method']:20s}"
            backbone_str = row['backbone'].split('/')[-1][:25]
            shots_str = f"{int(row['shots']):2d}-shot"
            
            print(f"{method_str} | {backbone_str:25s} | {shots_str} | "
                  f"{mean:5.2f}% ± {std:4.2f}% | 95% CI: [{ci_low:5.2f}%, {ci_high:5.2f}%]")
    
    print("\n" + "="*80 + "\n")


def statistical_comparison(df: pd.DataFrame, method1: str, method2: str,
                          dataset: str, shots: int) -> Dict:
    """Perform statistical comparison between two methods.
    
    Args:
        df: Results DataFrame
        method1: First method name
        method2: Second method name
        dataset: Dataset name
        shots: Number of shots
        
    Returns:
        Dictionary with comparison results
    """
    subset1 = df[(df['method'] == method1) & 
                 (df['dataset'] == dataset) & 
                 (df['shots'] == shots)]['val_acc'].values
    
    subset2 = df[(df['method'] == method2) & 
                 (df['dataset'] == dataset) & 
                 (df['shots'] == shots)]['val_acc'].values
    
    if len(subset1) == 0 or len(subset2) == 0:
        return {'error': 'Insufficient data for comparison'}
    
    # Paired t-test if same number of seeds
    if len(subset1) == len(subset2):
        t_stat, p_value = stats.ttest_rel(subset1, subset2)
        test_type = "Paired t-test"
    else:
        t_stat, p_value = stats.ttest_ind(subset1, subset2)
        test_type = "Independent t-test"
    
    mean_diff = np.mean(subset1) - np.mean(subset2)
    
    return {
        'test_type': test_type,
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'mean_difference': mean_diff * 100,
        f'{method1}_mean': np.mean(subset1) * 100,
        f'{method2}_mean': np.mean(subset2) * 100
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument('--results', type=str, default='./results/results.csv',
                       help='Path to results CSV file')
    parser.add_argument('--output', type=str, default='./results/aggregated_results.csv',
                       help='Path to save aggregated results')
    parser.add_argument('--metric', type=str, default='val_acc',
                       choices=['val_acc', 'val_f1'],
                       help='Metric to display in table')
    args = parser.parse_args()
    
    # Load and aggregate results
    print(f"Loading results from {args.results}...")
    aggregated = aggregate_results(args.results)
    
    # Save aggregated results
    aggregated.to_csv(args.output, index=False)
    print(f"Aggregated results saved to {args.output}")
    
    # Print formatted table
    print_results_table(aggregated, args.metric)
    
    # Statistical comparisons
    df = pd.read_csv(args.results)
    
    print("\nStatistical Comparisons:")
    print("="*80)
    
    # Compare Spectral-LoRA vs LoRA
    if 'spectral_lora' in df['method'].values and 'lora' in df['method'].values:
        for dataset in df['dataset'].unique():
            for shots in df['shots'].unique():
                comp = statistical_comparison(df, 'spectral_lora', 'lora', 
                                             dataset, shots)
                if 'error' not in comp:
                    sig_marker = "***" if comp['p_value'] < 0.001 else \
                                "**" if comp['p_value'] < 0.01 else \
                                "*" if comp['p_value'] < 0.05 else "ns"
                    
                    print(f"\n{dataset} ({shots}-shot): Spectral-LoRA vs LoRA")
                    print(f"  Spectral-LoRA: {comp['spectral_lora_mean']:.2f}%")
                    print(f"  LoRA: {comp['lora_mean']:.2f}%")
                    print(f"  Difference: {comp['mean_difference']:+.2f}%")
                    print(f"  p-value: {comp['p_value']:.4f} {sig_marker}")
    
    print("\n" + "="*80)
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns p>=0.05")


if __name__ == '__main__':
    main()
