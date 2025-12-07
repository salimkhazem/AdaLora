import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse
from pathlib import Path
import glob

# Style IEEE
sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
plt.rcParams['font.family'] = 'serif'

def get_singular_values_from_checkpoint(ckpt_path):
    """
    Extracts LoRA weights from a checkpoint and computes singular values.
    Navigates the state_dict to find lora_A and lora_B.
    """
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    svd_values = []
    
    # We iterate over the state dict to find pairs of lora_A and lora_B
    # This is robust to different architectures (ViT, CLIP, etc.)
    keys = list(state_dict.keys())
    lora_a_keys = [k for k in keys if 'lora_A' in k]
    
    print(f"Found {len(lora_a_keys)} LoRA layers.")
    
    # Aggregate SVD from ALL layers (or just take the middle one)
    # Taking the middle layer (e.g. layer 6 or 11) is standard for visualization
    target_layer_idx = len(lora_a_keys) // 2
    target_key_A = lora_a_keys[target_layer_idx]
    target_key_B = target_key_A.replace('lora_A', 'lora_B')
    
    if target_key_B not in state_dict:
        print(f"Error: Matching B key not found for {target_key_A}")
        return None

    A = state_dict[target_key_A].float().numpy()
    B = state_dict[target_key_B].float().numpy()
    
    # Compute Effective Weight Update: Delta W = B @ A
    # A shape: (rank, dim_in), B shape: (dim_out, rank)
    # W_delta shape: (dim_out, dim_in)
    
    W_delta = B @ A
    
    # Compute SVD
    # We only care about the top 'rank' singular values
    # But since B@A is rank 'r', we just compute SVD of B@A
    try:
        _, S, _ = np.linalg.svd(W_delta, full_matrices=False)
    except Exception as e:
        print(f"SVD Failed: {e}")
        return None
        
    return S

def plot_svd_comparison(lora_path, spectral_path, output_path, rank=8):
    """Plots the spectrum comparison."""
    s_lora = get_singular_values_from_checkpoint(lora_path)
    s_spectral = get_singular_values_from_checkpoint(spectral_path)
    
    if s_lora is None or s_spectral is None:
        return

    # Normalize usually helps visualization: S / max(S)
    s_lora = s_lora / s_lora[0]
    s_spectral = s_spectral / s_spectral[0]
    
    # Take only top 'rank' values to show the collapse clearly
    # Or show all up to 'rank'
    params_count = min(len(s_lora), len(s_spectral), rank * 2)
    x = np.arange(1, params_count + 1)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(x, s_lora[:params_count], label='Standard LoRA', color='#e74c3c', 
            marker='s', linewidth=2.5, linestyle='--')
    ax.plot(x, s_spectral[:params_count], label='Spectral-LoRA (Ours)', color='#2ecc71', 
            marker='o', linewidth=2.5)
    
    # Fill area to highlight difference
    ax.fill_between(x, s_lora[:params_count], s_spectral[:params_count], 
                    color='#2ecc71', alpha=0.1, label='Recovered Information')

    ax.set_xlabel('Singular Value Index', fontweight='bold')
    ax.set_ylabel('Normalized Magnitude', fontweight='bold')
    ax.set_title('Singular Value Spectrum (Layer 6)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.text(rank//2, 0.2, 'Rank Collapse\n(Standard LoRA)', color='#e74c3c', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved SVD Plot to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="./results", help="Directory containing .pt checkpoints")
    parser.add_argument("--dataset", required=True, help="eurosat, fgvc_aircraft, etc.")
    parser.add_argument("--shots", default=16, type=int)
    args = parser.parse_args()
    
    # Find best models automatically
    # Pattern: best_model_{dataset}_{method}_shot{shots}_seed*.pt
    # We take the first seed found for simplicity
    
    def find_ckpt(method):
        pattern = os.path.join(args.results_dir, f"best_model_{args.dataset}_{method}_shot{args.shots}_seed*.pt")
        files = glob.glob(pattern)
        return files[0] if files else None
    
    lora_ckpt = find_ckpt("lora")
    spectral_ckpt = find_ckpt("spectral_lora")
    
    if not lora_ckpt:
        print(f"Could not find LoRA checkpoint for {args.dataset}")
    if not spectral_ckpt:
        print(f"Could not find Spectral-LoRA checkpoint for {args.dataset}")
        
    if lora_ckpt and spectral_ckpt:
        print(f"Comparing:\n1. {lora_ckpt}\n2. {spectral_ckpt}")
        plot_svd_comparison(lora_ckpt, spectral_ckpt, f"./results/plots/{args.dataset}_svd_analysis.png")
    else:
        print("Skipping plot due to missing checkpoints.")

if __name__ == "__main__":
    main()
