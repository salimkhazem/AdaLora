"""
Simple AdaLoRA hyperparameter tuning script.
Tests different init_lambda values to find optimal configuration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from data import get_dataset, get_dataloader
from models_adalora import AdaLoRASpectral
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

def train_adalora(dataset, backbone, init_lambda, seed, shots=16, epochs=50):
    """Train AdaLoRA with specific hyperparameters."""
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load data
    train_set, val_set = get_dataset(dataset, './data', backbone)
    
    # Get num_classes - handle both Subset and direct datasets
    if hasattr(train_set, 'dataset'):
        # It's a Subset
        base_dataset = train_set.dataset
    else:
        # It's the dataset directly
        base_dataset = train_set
    
    # Get number of classes
    if hasattr(base_dataset, 'classes'):
        num_classes = len(base_dataset.classes)
    elif hasattr(base_dataset, '_labels'):
        num_classes = len(set(base_dataset._labels))
    else:
        num_classes = len(set(train_set.targets))
    train_loader = get_dataloader(train_set, 32, k_shots=shots, seed=seed, is_train=True)
    val_loader = get_dataloader(val_set, 32, is_train=False)
    
    # Create model
    print(f"\nTraining AdaLoRA: {dataset}, λ_init={init_lambda}, seed={seed}")
    model = AdaLoRASpectral(backbone, num_classes, rank=8, init_lambda=init_lambda, learn_lambda=True)
    model = model.cuda()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    best_f1 = 0
    best_epoch = 0
    lambda_history = []
    
    # Training
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_reg_loss = 0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            inputs, targets = inputs.cuda(), targets.cuda()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Add adaptive regularization
            reg_loss = model.compute_regularization()
            total_loss = loss + reg_loss
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_reg_loss += reg_loss.item()
        
        # Get lambda stats
        lambda_stats = model.get_lambda_stats()
        lambda_history.append(lambda_stats['mean'])
        
        # Validation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.cuda()
                outputs = model(inputs)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets.numpy())
        
        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='macro')
        
        if acc > best_acc:
            best_acc = acc
            best_f1 = f1
            best_epoch = epoch
            
            # Save checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'lambda_stats': lambda_stats,
                'lambda_history': lambda_history,
            }, f'./results/best_model_{dataset}_adalora_init{init_lambda}_shot{shots}_seed{seed}.pt')
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Acc={acc*100:.2f}%, λ_mean={lambda_stats['mean']:.4f}")
    
    print(f"Best: {best_acc*100:.2f}% (epoch {best_epoch}), Final λ={lambda_stats['mean']:.4f}")
    
    return {
        'dataset': dataset,
        'backbone': backbone,
        'init_lambda': init_lambda,
        'seed': seed,
        'shots': shots,
        'best_epoch': best_epoch,
        'val_acc': best_acc,
        'val_f1': best_f1,
        'learned_lambda_mean': lambda_stats['mean'],
        'learned_lambda_std': lambda_stats['std'],
        'learned_lambda_min': lambda_stats['min'],
        'learned_lambda_max': lambda_stats['max'],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='eurosat', choices=['eurosat', 'fgvc_aircraft'])
    parser.add_argument('--backbone', default='openai/clip-vit-base-patch32')
    parser.add_argument('--init_lambdas', nargs='+', type=float, default=[0.0001, 0.001, 0.01])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--shots', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    
    all_results = []
    
    print("="*80)
    print("AdaLoRA Hyperparameter Tuning")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Init lambdas: {args.init_lambdas}")
    print(f"Seeds: {args.seeds}")
    print(f"Total experiments: {len(args.init_lambdas) * len(args.seeds)}")
    print("="*80)
    
    for init_lambda in args.init_lambdas:
        for seed in args.seeds:
            result = train_adalora(
                args.dataset,
                args.backbone,
                init_lambda,
                seed,
                args.shots,
                args.epochs
            )
            all_results.append(result)
    
    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(f'./results/adalora_tuning_{args.dataset}.csv', index=False)
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    for init_lambda in args.init_lambdas:
        subset = df[df['init_lambda'] == init_lambda]
        mean_acc = subset['val_acc'].mean() * 100
        std_acc = subset['val_acc'].std() * 100
        mean_lambda = subset['learned_lambda_mean'].mean()
        
        print(f"\ninit_lambda = {init_lambda}:")
        print(f"  Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
        print(f"  Learned λ: {mean_lambda:.4f}")
    
    best_config = df.loc[df['val_acc'].idxmax()]
    print(f"\n✅ Best Configuration:")
    print(f"  Init Lambda: {best_config['init_lambda']}")
    print(f"  Accuracy: {best_config['val_acc']*100:.2f}%")
    print(f"  Learned Lambda: {best_config['learned_lambda_mean']:.4f}")
    
    print(f"\nResults saved to: ./results/adalora_tuning_{args.dataset}.csv")
