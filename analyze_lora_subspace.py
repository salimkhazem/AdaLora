"""
Comprehensive LoRA Subspace Analysis Tool

Analyzes the geometric properties of learned LoRA matrices to understand:
1. Singular value spectrum
2. Orthogonality of learned subspaces
3. Effective rank and condition number
4. Correlation with task performance

Senior Research Scientist Implementation
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import argparse
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Tuple
import json

# IEEE-style plotting
sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'


def compute_orthogonality_score(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute orthogonality score: how close A and B are to orthogonal.
    
    Score = ||AA^T - I||_F + ||B^TB - I||_F
    
    Lower is more orthogonal (0 = perfect orthogonality).
    """
    r = A.shape[0]
    I = np.eye(r)
    
    # A is (r, d_in), A @ A.T is (r, r)
    AAT = A @ A.T
    orthog_A = np.linalg.norm(AAT - I, 'fro')
    
    # B is (d_out, r), B.T @ B is (r, r)
    BTB = B.T @ B
    orthog_B = np.linalg.norm(BTB - I, 'fro')
    
    return orthog_A + orthog_B


def compute_effective_rank(singular_values: np.ndarray) -> float:
    """
    Compute effective rank: measure of how many singular values are significant.
    
    eff_rank = (Σ s_i)^2 / (Σ s_i^2)
    
    Equals full rank for uniform spectrum, lower for concentrated spectrum.
    """
    s_normalized = singular_values / singular_values.sum()
    return 1.0 / (s_normalized ** 2).sum()


def analyze_lora_checkpoint(ckpt_path: str, device='cpu') -> Dict:
    """
    Extract and analyze LoRA matrices from a checkpoint.
    
    Returns comprehensive metrics for all LoRA layers.
    """
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
    except Exception as e:
        print(f"Error loading {ckpt_path}: {e}")
        return None
    
    # Find all LoRA layers
    lora_a_keys = [k for k in state_dict.keys() if 'lora_A' in k and 'weight' in k]
    
    if not lora_a_keys:
        print(f"No LoRA layers found in {ckpt_path}")
        return None
    
    metrics_per_layer = []
    
    for key_a in lora_a_keys:
        key_b = key_a.replace('lora_A', 'lora_B')
        
        if key_b not in state_dict:
            continue
        
        # Extract matrices
        A = state_dict[key_a].float().cpu().numpy()  # (r, d_in)
        B = state_dict[key_b].float().cpu().numpy()  # (d_out, r)
        
        # Compute W_delta = B @ A
        W_delta = B @ A  # (d_out, d_in)
        
        # SVD of effective weight update
        try:
            U, S, Vt = np.linalg.svd(W_delta, full_matrices=False)
        except:
            print(f"SVD failed for {key_a}")
            continue
        
        # Compute metrics
        rank = A.shape[0]
        orthog_score = compute_orthogonality_score(A, B)
        eff_rank = compute_effective_rank(S[:rank])
        condition_number = S[0] / (S[rank-1] + 1e-10)
        
        # Spectral decay
        spectral_decay = S[1] / (S[0] + 1e-10)  # How quickly spectrum decays
        
        metrics_per_layer.append({
            'layer_name': key_a,
            'rank': rank,
            'orthogonality': orthog_score,
            'effective_rank': eff_rank,
            'condition_number': condition_number,
            'spectral_decay': spectral_decay,
            'top_singular_value': S[0],
            'singular_values': S[:rank].tolist(),
        })
    
    # Aggregate metrics
    if not metrics_per_layer:
        return None
    
    aggregated = {
        'checkpoint_path': ckpt_path,
        'num_lora_layers': len(metrics_per_layer),
        'avg_orthogonality': np.mean([m['orthogonality'] for m in metrics_per_layer]),
        'std_orthogonality': np.std([m['orthogonality'] for m in metrics_per_layer]),
        'avg_effective_rank': np.mean([m['effective_rank'] for m in metrics_per_layer]),
        'avg_condition_number': np.mean([m['condition_number'] for m in metrics_per_layer]),
        'avg_spectral_decay': np.mean([m['spectral_decay'] for m in metrics_per_layer]),
        'layer_metrics': metrics_per_layer,
    }
    
    return aggregated


def extract_metadata_from_path(ckpt_path: str) -> Dict:
    """Extract experiment metadata from checkpoint filename."""
    # Pattern: best_model_{dataset}_{method}_shot{shots}_seed{seed}.pt
    filename = Path(ckpt_path).stem
    parts = filename.split('_')
    
    try:
        metadata = {
            'dataset': parts[2],
            'method': parts[3],
            'shots': int(parts[4].replace('shot', '')),
            'seed': int(parts[5].replace('seed', '')),
        }
        return metadata
    except:
        return {}


def analyze_all_checkpoints(results_dir: str, output_csv: str):
    """
    Analyze all checkpoints in results directory.
    
    Saves comprehensive CSV with all metrics.
    """
    checkpoint_pattern = str(Path(results_dir) / 'best_model_*.pt')
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    print(f"Found {len(checkpoint_files)} checkpoints to analyze")
    
    all_results = []
    
    for i, ckpt_path in enumerate(checkpoint_files):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(checkpoint_files)}")
        
        # Analyze checkpoint
        metrics = analyze_lora_checkpoint(ckpt_path)
        if metrics is None:
            continue
        
        # Extract metadata
        metadata = extract_metadata_from_path(ckpt_path)
        
        # Load results.csv to get performance
        # Combine metrics with metadata
        result_row = {
            **metadata,
            **metrics
        }
        
        all_results.append(result_row)
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved analysis to {output_csv}")
    
    return df


def correlate_with_performance(analysis_csv: str, results_csv: str, output_dir: str):
    """
    Correlate subspace metrics with task performance.
    
    Generates correlation plots and statistics.
    """
    # Load data
    df_analysis = pd.read_csv(analysis_csv)
    df_results = pd.read_csv(results_csv)
    
    # Merge on metadata
    df_merged = df_analysis.merge(
        df_results[['dataset', 'method', 'shots', 'seed', 'val_acc']],
        on=['dataset', 'method', 'shots', 'seed'],
        how='inner'
    )
    
    print(f"Merged {len(df_merged)} checkpoints with performance data")
    
    # Correlation analysis
    metrics_to_correlate = [
        'avg_orthogonality',
        'avg_effective_rank',
        'avg_condition_number',
        'avg_spectral_decay'
    ]
    
    correlations = {}
    for metric in metrics_to_correlate:
        if metric in df_merged.columns:
            corr, p_value = pearsonr(df_merged[metric], df_merged['val_acc'])
            correlations[metric] = {
                'pearson_r': corr,
                'p_value': p_value
            }
            print(f"{metric}: r={corr:.3f}, p={p_value:.4f}")
    
    # Save correlations
    with open(Path(output_dir) / 'correlations.json', 'w') as f:
        json.dump(correlations, f, indent=2)
    
    return df_merged, correlations


def main():
    parser = argparse.ArgumentParser(description="LoRA Subspace Analysis")
    parser.add_argument('--results_dir', default='./results', 
                       help='Directory containing checkpoints')
    parser.add_argument('--output_dir', default='./results/subspace_analysis',
                       help='Output directory for analysis')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("LoRA SUBSPACE ANALYSIS")
    print("="*80)
    
    # Step 1: Analyze all checkpoints
    print("\n[1/3] Analyzing checkpoints...")
    analysis_csv = output_dir / 'subspace_metrics.csv'
    df_analysis = analyze_all_checkpoints(args.results_dir, str(analysis_csv))
    
    # Step 2: Correlate with performance
    print("\n[2/3] Correlating with performance...")
    results_csv = Path(args.results_dir) / 'results.csv'
    if results_csv.exists():
        df_merged, correlations = correlate_with_performance(
            str(analysis_csv), 
            str(results_csv),
            str(output_dir)
        )
    else:
        print(f"Results CSV not found at {results_csv}")
        df_merged = df_analysis
    
    # Step 3: Summary statistics
    print("\n[3/3] Computing summary statistics...")
    
    # Group by backbone and method
    if 'method' in df_analysis.columns:
        summary = df_analysis.groupby(['method']).agg({
            'avg_orthogonality': ['mean', 'std'],
            'avg_effective_rank': ['mean', 'std'],
            'avg_condition_number': ['mean', 'std'],
        }).round(3)
        
        print("\nSummary by Method:")
        print(summary)
        
        summary.to_csv(output_dir / 'summary_statistics.csv')
    
    print(f"\n✅ Analysis complete! Results in {output_dir}")
    print(f"   - subspace_metrics.csv: Per-checkpoint metrics")
    print(f"   - correlations.json: Correlation with performance")
    print(f"   - summary_statistics.csv: Aggregated statistics")


if __name__ == '__main__':
    main()
