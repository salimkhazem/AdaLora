"""
Utility functions for experiment management and analysis.
"""
from typing import Dict, List, Tuple, Optional
import os
import json
import torch
import numpy as np
from pathlib import Path


def save_config(args, output_dir: str) -> None:
    """Save experiment configuration to JSON file.
    
    Args:
        args: Parsed arguments from argparse
        output_dir: Directory to save configuration
    """
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to {config_path}")


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, 
                    optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
    """Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        
    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def compute_confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Compute confidence interval for a list of values.
    
    Args:
        values: List of numerical values
        confidence: Confidence level (default: 0.95)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    from scipy import stats
    
    values = np.array(values)
    mean = np.mean(values)
    std_err = stats.sem(values)
    margin = std_err * stats.t.ppf((1 + confidence) / 2, len(values) - 1)
    
    return mean - margin, mean + margin


def format_metric_with_ci(values: List[float], metric_name: str = "Accuracy") -> str:
    """Format metric with mean ± std and confidence interval.
    
    Args:
        values: List of metric values across seeds
        metric_name: Name of the metric
        
    Returns:
        Formatted string with statistics
    """
    mean = np.mean(values)
    std = np.std(values)
    ci_low, ci_high = compute_confidence_interval(values)
    
    return f"{metric_name}: {mean:.2f}% ± {std:.2f}% (95% CI: [{ci_low:.2f}%, {ci_high:.2f}%])"
