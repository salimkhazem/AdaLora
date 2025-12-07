"""
Visualization Suite for LoRA Subspace Analysis

Creates publication-quality figures showing:
1. SVD spectrum comparison
2. Orthogonality vs performance scatter
3. Backbone comparison heatmaps
4. Layer-wise analysis

Senior Research Scientist Implementation - IEEE/ICPR Style
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
import json

# Publication-quality settings
sns.set_theme(style="white", context="paper")
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'text.usetex': False,  # Set True if LaTeX installed
})


def plot_svd_spectrum(df: pd.DataFrame, output_path: str):
    """
    Plot SVD spectrum comparison: LoRA vs Spectral-LoRA.
    
    Shows rank collapse in standard LoRA vs maintained spectrum in Spectral-LoRA.
    """
    # Filter for specific condition
    df_plot = df[
        (df['dataset'] == 'eurosat') &
        (df['shots'] == 16) &
        (df['seed'] == 42)
    ]
    
    if len(df_plot) == 0:
        print("No data for SVD spectrum plot")
        return
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    colors = {'lora': '#e74c3c', 'spectral_lora': '#2ecc71', 'linear': '#3498db'}
    markers = {'lora': 's', 'spectral_lora': 'o', 'linear': '^'}
    
    for method in ['lora', 'spectral_lora']:
        method_data = df_plot[df_plot['method'] == method]
        
        if len(method_data) == 0:
            continue
        
        # Extract singular values from first checkpoint
        sv_list = eval(method_data.iloc[0]['layer_metrics'])[6]['singular_values']  # Middle layer
        sv_normalized = np.array(sv_list) / sv_list[0]
        
        x = np.arange(1, len(sv_normalized) + 1)
        
        label = 'Standard LoRA' if method == 'lora' else 'Spectral-LoRA'
        
        ax.plot(x, sv_normalized, 
               label=label,
               color=colors[method],
               marker=markers[method],
               linewidth=2.5,
               markersize=7,
               markevery=1)
    
    ax.set_xlabel('Singular Value Index', fontweight='bold')
    ax.set_ylabel('Normalized Magnitude', fontweight='bold')
    ax.set_title('Singular Value Spectrum (EuroSAT, 16-shot)', fontweight='bold')
    ax.legend(frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_orthogonality_vs_performance(df_merged: pd.DataFrame, output_path: str):
    """
    Scatter plot: Orthogonality score vs validation accuracy.
    
    Shows correlation between geometric properties and performance.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Color by backbone
    backbones = df_merged['dataset'].unique()
    palette = sns.color_palette("husl", n_colors=len(backbones))
    
    for i, dataset in enumerate(backbones):
        data = df_merged[df_merged['dataset'] == dataset]
        
        ax.scatter(
            data['avg_orthogonality'],
            data['val_acc'] * 100,
            label=dataset.upper(),
            alpha=0.6,
            s=80,
            color=palette[i],
            edgecolors='black',
            linewidth=0.5
        )
    
    # Add trend line
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(
        df_merged['avg_orthogonality'],
        df_merged['val_acc'] * 100
    )
    
    x_line = np.linspace(df_merged['avg_orthogonality'].min(), 
                        df_merged['avg_orthogonality'].max(), 100)
    y_line = slope * x_line + intercept
    
    ax.plot(x_line, y_line, 'k--', linewidth=2, alpha=0.5,
           label=f'Linear fit (r={r_value:.3f}, p={p_value:.4f})')
    
    ax.set_xlabel('Average Orthogonality Score', fontweight='bold')
    ax.set_ylabel('Validation Accuracy (%)', fontweight='bold')
    ax.set_title('Orthogonality vs Performance', fontweight='bold')
    ax.legend(frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_method_comparison_heatmap(df: pd.DataFrame, output_path: str):
    """
    Heatmap comparing methods across different metrics.
    """
    if 'method' not in df.columns:
        print("No method column for heatmap")
        return
    
    # pivot table
    metrics = ['avg_orthogonality', 'avg_effective_rank', 'avg_condition_number']
    
    # Compute mean for each method
    method_stats = df.groupby('method')[metrics].mean()
    
    # Normalize for easier comparison (z-score)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    method_stats_norm = pd.DataFrame(
        scaler.fit_transform(method_stats),
        index=method_stats.index,
        columns=method_stats.columns
    )
    
    # Rename for readability
    method_stats_norm.columns = ['Orthogonality', 'Effective Rank', 'Condition Number']
    method_stats_norm.index = [m.replace('_', ' ').title() for m in method_stats_norm.index]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    sns.heatmap(method_stats_norm, annot=True, fmt='.2f', cmap='RdYlGn_r',
               center=0, cbar_kws={'label': 'Normalized Score'},
               linewidths=1, linecolor='white', ax=ax)
    
    ax.set_title('Method Comparison: Subspace Metrics (Normalized)', fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_layer_analysis(df: pd.DataFrame, output_path: str):
    """
    Layer-wise analysis: How orthogonality changes with depth.
    """
    # Get one checkpoint of each method
    sample_checkpoints = df.groupby('method').first()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    colors = {'lora': '#e74c3c', 'spectral_lora': '#2ecc71'}
    
    for method in ['lora', 'spectral_lora']:
        if method not in sample_checkpoints.index:
            continue
        
        layer_metrics = eval(sample_checkpoints.loc[method, 'layer_metrics'])
        
        layers = range(len(layer_metrics))
        orthog_scores = [m['orthogonality'] for m in layer_metrics]
        eff_ranks = [m['effective_rank'] for m in layer_metrics]
        
        label = 'Standard LoRA' if method == 'lora' else 'Spectral-LoRA'
        
        ax1.plot(layers, orthog_scores, marker='o', linewidth=2,
                label=label, color=colors[method])
        
        ax2.plot(layers, eff_ranks, marker='s', linewidth=2,
                label=label, color=colors[method])
    
    ax1.set_xlabel('Layer Index', fontweight='bold')
    ax1.set_ylabel('Orthogonality Score', fontweight='bold')
    ax1.set_title('Orthogonality by Layer', fontweight='bold')
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Layer Index', fontweight='bold')
    ax2.set_ylabel('Effective Rank', fontweight='bold')
    ax2.set_title('Effective Rank by Layer', fontweight='bold')
    ax2.legend(frameon=True)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize LoRA Subspace Analysis")
    parser.add_argument('--analysis_dir', default='./results/subspace_analysis',
                       help='Directory with analysis results')
    parser.add_argument('--output_dir', default='./results/subspace_analysis/figures',
                       help='Output directory for figures')
    args = parser.parse_args()
    
    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("LORA SUBSPACE VISUALIZATION")
    print("="*80)
    
    # Load data
    metrics_csv = analysis_dir / 'subspace_metrics.csv'
    if not metrics_csv.exists():
        print(f"Error: {metrics_csv} not found. Run analyze_lora_subspace.py first")
        return
    
    df = pd.read_csv(metrics_csv)
    print(f"Loaded {len(df)} checkpoints")
    
    # Try to load merged data with performance
    merged_csv = analysis_dir / 'merged_metrics.csv'
    if merged_csv.exists():
        df_merged = pd.read_csv(merged_csv)
    else:
        df_merged = df
    
    # Generate all figures
    print("\n[1/4] SVD Spectrum...")
    plot_svd_spectrum(df, str(output_dir / 'svd_spectrum.png'))
    
    print("\n[2/4] Orthogonality vs Performance...")
    if 'val_acc' in df_merged.columns:
        plot_orthogonality_vs_performance(df_merged, 
                                         str(output_dir / 'orthog_vs_perf.png'))
    
    print("\n[3/4] Method Comparison Heatmap...")
    plot_method_comparison_heatmap(df, str(output_dir / 'method_comparison.png'))
    
    print("\n[4/4] Layer-wise Analysis...")
    plot_layer_analysis(df, str(output_dir / 'layer_analysis.png'))
    
    print(f"\n✅ All figures saved to {output_dir}")
    print("   Generated:")
    print("   - svd_spectrum.png/pdf")
    print("   - orthog_vs_perf.png/pdf")
    print("   - method_comparison.png/pdf")
    print("   - layer_analysis.png/pdf")


if __name__ == '__main__':
    main()
