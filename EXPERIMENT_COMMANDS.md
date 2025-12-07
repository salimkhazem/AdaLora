# QUICK START: Commands to Run Experiments

## IMMEDIATE COMMANDS

### Option 1: Run Everything at Once (Recommended)
```bash
chmod +x run_novel_methods.sh
./run_novel_methods.sh
```

**What this does**:
- Runs all 36 experiments (18 OrthInit + 18 AdaLoRA)
- Automatically aggregates results
- Generates all plots and visualizations
- Creates final analysis report
- **Time**: ~8-10 hours on 3 GPUs

---

### Option 2: Run Methods Separately

#### A) OrthInit-LoRA Only (18 experiments, ~4 hours)
```bash
chmod +x run_orthinit_experiments.sh
./run_orthinit_experiments.sh
```

Then analyze:
```bash
.venv/bin/python aggregate_results.py --results ./results/results.csv
.venv/bin/python visualize.py --results ./results/results.csv
```

#### B) AdaLoRA-S Only (create script first):
```bash
# Create AdaLoRA experiment script
cat > run_adalora_experiments.sh << 'EOF'
#!/bin/bash
set -e

RESULTS_DIR="./results"
DATA_ROOT="./data"
DATASETS=("eurosat" "fgvc_aircraft")
BACKBONES=("openai/clip-vit-base-patch32" "facebook/dinov2-base" "google/siglip-base-patch16-224")
SHOTS=16
SEEDS=(42 43 44)

counter=0
for BACKBONE in "${BACKBONES[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            counter=$((counter + 1))
            GPU=$((counter % 3))
            
            echo "[$counter/18] AdaLoRA: $DATASET | ${BACKBONE##*/} | Seed $SEED"
            
            CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python main.py \
                --dataset $DATASET \
                --data_root $DATA_ROOT \
                --backbone $BACKBONE \
                --method adalora \
                --shots $SHOTS \
                --epochs 50 \
                --seed $SEED \
                --output_dir $RESULTS_DIR \
                --no_wandb \
                --batch_size 32 \
                --lr 0.001 \
                --rank 8 &
            
            if [ $((counter % 3)) -eq 0 ]; then wait; fi
        done
    done
done
wait
echo "AdaLoRA experiments complete!"
EOF

chmod +x run_adalora_experiments.sh
./run_adalora_experiments.sh
```

Then analyze:
```bash
.venv/bin/python aggregate_results.py --results ./results/results.csv
.venv/bin/python visualize.py --results ./results/results.csv
```

---

### Option 3: Run Single Test Experiments (Quick validation)

#### Test OrthInit (1 experiment, ~15 min):
```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python main.py \
    --dataset eurosat \
    --data_root ./data \
    --backbone openai/clip-vit-base-patch32 \
    --method orthinit_lora \
    --shots 16 \
    --epochs 50 \
    --seed 42 \
    --output_dir ./results \
    --no_wandb \
    --batch_size 32 \
    --lr 0.001 \
    --rank 8
```

#### Test AdaLoRA (1 experiment, ~15 min):
```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python main.py \
    --dataset eurosat \
    --data_root ./data \
    --backbone openai/clip-vit-base-patch32 \
    --method adalora \
    --shots 16 \
    --epochs 50 \
    --seed 42 \
    --output_dir ./results \
    --no_wandb \
    --batch_size 32 \
    --lr 0.001 \
    --rank 8
```

---

## ANALYSIS COMMANDS (Run After Experiments)

### Full Analysis Pipeline:
```bash
# 1. Aggregate results
.venv/bin/python aggregate_results.py \
    --results ./results/results.csv \
    --output ./results/aggregated_results.csv

# 2. Generate plots
.venv/bin/python visualize.py \
    --results ./results/results.csv \
    --output_dir ./results/plots

# 3. Subspace analysis (analyze new checkpoints)
.venv/bin/python analyze_lora_subspace.py \
    --results_dir ./results \
    --output_dir ./results/subspace_analysis

# 4. Subspace visualizations
.venv/bin/python visualize_subspace.py \
    --analysis_dir ./results/subspace_analysis \
    --output_dir ./results/subspace_analysis/figures
```

---

## CHECK RESULTS

### View aggregated statistics:
```bash
head -20 results/aggregated_results.csv
```

### Count completed experiments:
```bash
wc -l results/results.csv
```

### Check specific method results:
```bash
grep "orthinit_lora" results/results.csv | wc -l  # Should be 18
grep "adalora" results/results.csv | wc -l         # Should be 18
```

### View latest experiments:
```bash
tail -20 results/results.csv
```

---

## OUTPUT FILES

After running experiments and analysis, you'll have:

```
results/
├── results.csv                          # Raw results (all experiments)
├── aggregated_results.csv               # Statistics (mean, std, CI)
├── best_model_*.pt                      # Saved checkpoints
├── plots/                               # Main visualizations
│   ├── eurosat_method_comparison.png
│   ├── fgvc_aircraft_method_comparison.png
│   ├── backbone_comparison.png
│   ├── eurosat_reg_ablation.png
│   └── fgvc_aircraft_reg_ablation.png
└── subspace_analysis/                   # Geometric analysis
    ├── subspace_metrics.csv
    ├── correlations.json
    ├── summary_statistics.csv
    └── figures/
        ├── svd_spectrum.png/pdf
        ├── orthog_vs_perf.png/pdf
        ├── method_comparison.png/pdf
        └── layer_analysis.png/pdf
```

---

## RECOMMENDED WORKFLOW

### For Quick Validation (30 min):
```bash
# Run 2 test experiments
CUDA_VISIBLE_DEVICES=0 .venv/bin/python main.py --dataset eurosat --method orthinit_lora --shots 16 --epochs 50 --seed 42 --no_wandb --output_dir ./results
CUDA_VISIBLE_DEVICES=0 .venv/bin/python main.py --dataset eurosat --method adalora --shots 16 --epochs 50 --seed 42 --no_wandb --output_dir ./results

# Quick check
tail results/results.csv
```

### For Full Evaluation (8-10 hours):
```bash
# Run everything
./run_novel_methods.sh

# Results automatically generated!
```

Choose Option 1 (run_novel_methods.sh) for complete automation!
