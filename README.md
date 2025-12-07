# Spectral-LoRA: Orthogonal Low-Rank Adaptation for Few-Shot Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/pytorch-2.5+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of **Spectral-LoRA** for ICPR 2026.

## Overview

Spectral-LoRA introduces a novel spectral regularization term that enforces orthogonality constraints on LoRA adaptation matrices, improving few-shot classification performance on vision transformers.

### Key Contributions

1. **Orthogonal LoRA Regularization**: Novel regularization term $L_{reg} = ||(AA^T) - I||_F^2 + ||(B^TB) - I||_F^2$
2. **Comprehensive Evaluation**: Extensive experiments on 3 datasets, 4 backbones, multiple shot settings
3. **Strong Baselines**: Comparison with Linear Probing and standard LoRA
4. **Thorough Ablations**: Analysis of regularization weight and rank

## Installation

### Requirements

- Python 3.8+
- CUDA 11.8+ (for GPU support)

### Setup

```bash
# Clone repository
git clone this repo
cd icpr_1

# Create virtual environment with uv (recommended)
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Dependencies

All dependencies are pinned to exact versions for reproducibility. See `requirements.txt`.

## Quick Start

### Single Experiment

```bash
python main.py \
    --dataset eurosat \
    --backbone openai/clip-vit-base-patch32 \
    --method spectral_lora \
    --shots 16 \
    --epochs 50 \
    --seed 42 \
    --no_wandb
```

### Reproduce Paper Results

Run all experiments from the paper:

```bash
./run_comprehensive_experiments.sh
```

This will:
- Run 3 methods (Linear, LoRA, Spectral-LoRA)
- On 3 datasets (EuroSAT, Stanford Cars, FGVC Aircraft)
- With 4 shot settings (1, 4, 8, 16)
- Using 4 backbones (CLIP, DINOv2, SigLIP, SigLIP-SO400M)
- With 3 random seeds each
- Plus ablation studies on regularization weight and rank

**Note**: Full experiments take approximately 48-72 hours on 3x A6000 GPUs.

## Repository Structure

```
spectral-lora/
├── main.py                           # Main training script
├── models.py                         # Model definitions & regularization
├── data.py                           # Dataset loaders & few-shot sampling
├── utils.py                          # Utility functions
├── aggregate_results.py              # Results aggregation & statistics
├── visualize.py                      # Visualization utilities
├── run_comprehensive_experiments.sh  # Full experiment suite
├── requirements.txt                  # Pinned dependencies
├── README.md                         # This file
└── results/                          # Experiment outputs (created automatically)
    ├── results.csv                   # Raw results
    ├── aggregated_results.csv        # Aggregated statistics
    └── plots/                        # Visualization plots
```

## Usage

### Arguments

Key command-line arguments:

```
--dataset           Dataset: {eurosat, stanford_cars, fgvc_aircraft}
--backbone          Model: {openai/clip-vit-base-patch32, facebook/dinov2-base, ...}
--method            Method: {linear, lora, spectral_lora}
--shots             Number of shots per class (default: 16)
--epochs            Training epochs (default: 50) 
--seed              Random seed (default: 42)
--rank              LoRA rank (default: 8)
--reg_weight        Spectral regularization weight λ (default: 0.1)
--lr                Learning rate (default: 1e-3)
--batch_size        Batch size (default: 32)
```

### Supported Backbones

- `openai/clip-vit-base-patch32` - CLIP ViT-B/32
- `facebook/dinov2-base` - DINOv2-base
- `google/siglip-base-patch16-224` - SigLIP-base
- `google/siglip-so400m-patch14-384` - SigLIP-SO400M

### Supported Datasets

- **EuroSAT**: 10-class satellite image classification
- **Stanford Cars**: 196-class fine-grained car recognition
- **FGVC Aircraft**: 100-class aircraft recognition

## Analysis

### Aggregate Results

```bash
python aggregate_results.py \
    --results ./results/results.csv \
    --output ./results/aggregated_results.csv
```

This computes:
- Mean ± standard deviation
- 95% confidence intervals
- Statistical significance tests (t-tests)

### Generate Plots

```bash
python visualize.py \
    --results ./results/results.csv \
    --output_dir ./results/plots
```

Creates:
- Method comparison plots
- Backbone comparison plots
- Ablation study plots
- Publication-ready PNG and PDF formats

## Reproducibility

We ensure full reproducibility through:

1. **Fixed Random Seeds**: All random operations are seeded
2. **Deterministic Algorithms**: `torch.backends.cudnn.deterministic = True`
3. **Pinned Dependencies**: Exact package versions in `requirements.txt`
4. **Saved Configurations**: All hyperparameters logged to `config.json`
5. **Model Checkpoints**: Best models saved with full training state

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/), [Hugging Face Transformers](https://huggingface.co/transformers/), and [PEFT](https://github.com/huggingface/peft)
- Datasets from torchvision and public repositories

