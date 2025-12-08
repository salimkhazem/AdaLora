#!/bin/bash
# Simple AdaLoRA Tuning Runner
# Runs hyperparameter search using tune_adalora.py

set -e

echo "========================================="
echo "AdaLoRA Hyperparameter Tuning"
echo "========================================="
echo ""

# Configuration
INIT_LAMBDAS="0.0001 0.001 0.01"
SEEDS="42 43 44"
EPOCHS=50
GPU=0

echo "Testing init_lambda values: $INIT_LAMBDAS"
echo "Seeds: $SEEDS"
echo "Epochs: $EPOCHS"
echo ""

# EuroSAT
echo "========================================="
echo "Running EuroSAT Experiments"
echo "========================================="
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python tune_adalora.py \
    --dataset eurosat \
    --init_lambdas $INIT_LAMBDAS \
    --seeds $SEEDS \
    --epochs $EPOCHS

echo ""
echo "✅ EuroSAT complete!"
echo ""

# FGVC Aircraft
echo "========================================="
echo "Running FGVC Aircraft Experiments"
echo "========================================="
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python tune_adalora.py \
    --dataset fgvc_aircraft \
    --init_lambdas $INIT_LAMBDAS \
    --seeds $SEEDS \
    --epochs $EPOCHS

echo ""
echo "✅ Aircraft complete!"
echo ""

# Summary
echo "========================================="
echo "🎉 ALL TUNING COMPLETE!"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  - results/adalora_tuning_eurosat.csv"
echo "  - results/adalora_tuning_fgvc_aircraft.csv"
echo ""
echo "Check logs above for best configurations!"
