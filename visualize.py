"""
Visualization utilities for experiment results.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
from typing import List, Optional


sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_method_comparison(df: pd.DataFrame, dataset: str, 
                           output_dir: str = './results/plots') -> None:
    """Plot comparison of methods across different shot counts.
    
    Args:
        df: Results DataFrame
        dataset: Dataset name to plot
        output_dir: Directory to save plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Filter for specific dataset
    df_dataset = df[df['dataset'] == dataset].copy()
    
    # Aggregate by method and shots
    agg = df_dataset.groupby(['method', 'shots']).agg({
        'val_acc': ['mean', 'std']
    }).reset_index()
    agg.columns = ['method', 'shots', 'acc_mean', 'acc_std']
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method in agg['method'].unique():
        method_data = agg[agg['method'] == method]
        x = method_data['shots']
        y = method_data['acc_mean'] * 100
        yerr = method_data['acc_std'] * 100
        
        ax.errorbar(x, y, yerr=yerr, marker='o', capsize=5, 
                   label=method.replace('_', ' ').title(), linewidth=2)
    
    ax.set_xlabel('Number of Shots', fontsize=12)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title(f'{dataset.upper()} - Method Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset}_method_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/{dataset}_method_comparison.pdf', bbox_inches='tight')
    print(f"Saved plot: {output_dir}/{dataset}_method_comparison.png")
    plt.close()


def plot_backbone_comparison(df: pd.DataFrame, output_dir: str = './results/plots') -> None:
    """Plot comparison across different backbones.
    
    Args:
        df: Results DataFrame
        output_dir: Directory to save plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Aggregate by backbone and dataset
    agg = df.groupby(['backbone', 'dataset', 'method']).agg({
        'val_acc': ['mean', 'std']
    }).reset_index()
    agg.columns = ['backbone', 'dataset', 'method', 'acc_mean', 'acc_std']
    
    # Simplify backbone names
    agg['backbone_short'] = agg['backbone'].apply(lambda x: x.split('/')[-1][:20])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    datasets = agg['dataset'].unique()
    x = np.arange(len(datasets))
    width = 0.25
    
    backbones = agg['backbone_short'].unique()
    for i, backbone in enumerate(backbones):
        backbone_data = agg[agg['backbone_short'] == backbone]
        means = [backbone_data[backbone_data['dataset'] == d]['acc_mean'].values[0] * 100 
                if len(backbone_data[backbone_data['dataset'] == d]) > 0 else 0
                for d in datasets]
        stds = [backbone_data[backbone_data['dataset'] == d]['acc_std'].values[0] * 100
               if len(backbone_data[backbone_data['dataset'] == d]) > 0 else 0
               for d in datasets]
        
        ax.bar(x + i * width, means, width, yerr=stds, 
              label=backbone, capsize=4)
    
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('Backbone Comparison Across Datasets', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([d.upper() for d in datasets])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/backbone_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/backbone_comparison.pdf', bbox_inches='tight')
    print(f"Saved plot: {output_dir}/backbone_comparison.png")
    plt.close()


def plot_regularization_ablation(df: pd.DataFrame, dataset: str,
                                 output_dir: str = './results/plots') -> None:
    """Plot ablation study of regularization weight.
    
    Args:
        df: Results DataFrame
        dataset: Dataset name
        output_dir: Directory to save plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Filter for spectral_lora method
    df_ablation = df[(df['dataset'] == dataset) & 
                     (df['method'] == 'spectral_lora')].copy()
    
    if 'reg_weight' not in df_ablation.columns or df_ablation.empty:
        print(f"No regularization ablation data for {dataset}")
        return
    
    # Aggregate
    agg = df_ablation.groupby('reg_weight').agg({
        'val_acc': ['mean', 'std']
    }).reset_index()
    agg.columns = ['reg_weight', 'acc_mean', 'acc_std']
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.errorbar(agg['reg_weight'], agg['acc_mean'] * 100, 
               yerr=agg['acc_std'] * 100, marker='o', capsize=5, linewidth=2)
    
    ax.set_xlabel('Regularization Weight (λ)', fontsize=12)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title(f'{dataset.upper()} - Regularization Weight Ablation', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset}_reg_ablation.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/{dataset}_reg_ablation.pdf', bbox_inches='tight')
    print(f"Saved plot: {output_dir}/{dataset}_reg_ablation.png")
    plt.close()


def create_all_plots(results_csv: str, output_dir: str = './results/plots') -> None:
    """Create all visualization plots.
    
    Args:
        results_csv: Path to results CSV
        output_dir: Directory to save plots
    """
    df = pd.read_csv(results_csv)
    
    print("Creating visualizations...")
    
    # Method comparison for each dataset
    for dataset in df['dataset'].unique():
        plot_method_comparison(df, dataset, output_dir)
    
    # Backbone comparison
    if len(df['backbone'].unique()) > 1:
        plot_backbone_comparison(df, output_dir)
    
    # Regularization ablation
    for dataset in df['dataset'].unique():
        plot_regularization_ablation(df, dataset, output_dir)
    
    print(f"\nAll plots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Create visualization plots")
    parser.add_argument('--results', type=str, default='./results/results.csv',
                       help='Path to results CSV file')
    parser.add_argument('--output_dir', type=str, default='./results/plots',
                       help='Directory to save plots')
    args = parser.parse_args()
    
    create_all_plots(args.results, args.output_dir)


if __name__ == '__main__':
    main()
