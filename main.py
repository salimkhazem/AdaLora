import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
import wandb
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

from data import get_dataset, get_dataloader
from models import get_model, spectral_regularization_loss

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def main():
    parser = argparse.ArgumentParser(description="Spectral-LoRA Experiments")
    parser.add_argument("--dataset", type=str, required=True, choices=['stanford_cars', 'fgvc_aircraft', 'eurosat'])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--shots", type=int, default=16, help="Number of shots per class")
    parser.add_argument("--backbone", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--method", type=str, default="lora", 
                       choices=['linear', 'lora', 'spectral_lora', 'orthinit_lora', 'adalora'])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--reg_weight", type=float, default=0.1, help="Weight for spectral regularization")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--wandb_project", type=str, default="spectral-lora-icpr")
    parser.add_argument("--no_wandb", action="store_true")
    
    args = parser.parse_args()
    
    # Initialize Accelerator
    accelerator = Accelerator()
    device = accelerator.device
    
    set_seed(args.seed)
    
    # Logging
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        if not args.no_wandb:
            wandb.init(project=args.wandb_project, config=vars(args))
            
    # Data
    accelerator.print(f"Loading dataset: {args.dataset}")
    train_set, val_set = get_dataset(args.dataset, args.data_root, args.backbone)
    
    # Determine num classes
    if hasattr(train_set, 'classes'):
        num_classes = len(train_set.classes)
    elif hasattr(train_set, 'dataset') and hasattr(train_set.dataset, 'classes'): # Subset
        num_classes = len(train_set.dataset.classes)
    else:
        # Fallback: iterate or check targets max
        # This might be slow for large datasets but fine for these
        # Assume targets are 0..N-1
        # For EuroSAT/FGVC/Cars this is usually true.
        # Let's try to find a safer way or assume standard counts.
        if args.dataset == 'stanford_cars': num_classes = 196
        elif args.dataset == 'fgvc_aircraft': num_classes = 100
        elif args.dataset == 'eurosat': num_classes = 10
        else: num_classes = 100 # Default fallback
        
    accelerator.print(f"Num classes: {num_classes}")

    train_loader = get_dataloader(train_set, args.batch_size, k_shots=args.shots, seed=args.seed, is_train=True)
    val_loader = get_dataloader(val_set, args.batch_size, is_train=False)
    
    # Model
    accelerator.print(f"Loading model: {args.backbone} ({args.method})")
    
    if args.method == 'adalora':
        # Import AdaLoRA model
        from models_adalora import AdaLoRASpectral
        model = AdaLoRASpectral(args.backbone, num_classes, rank=args.rank)
    else:
        model = get_model(args.backbone, args.method, num_classes, args.rank)
    
    # Optimizer
    # Only optimize trainable parameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr)
    
    criterion = nn.CrossEntropyLoss()
    
    # Prepare with Accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    # Training Loop
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_reg_loss = 0.0
        
        for batch in tqdm(train_loader, disable=not accelerator.is_local_main_process, desc=f"Epoch {epoch+1}/{args.epochs}"):
            inputs, targets = batch
            # inputs = inputs.to(device) # Handled by accelerator
            # targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Add regularization if applicable
            reg_loss = torch.tensor(0.0, device=device) # Initialize reg_loss
            if args.method == 'spectral_lora':
                reg_loss = spectral_regularization_loss(model)
                loss_to_backward = loss + args.reg_weight * reg_loss
                train_reg_loss += reg_loss.item()
            elif args.method == 'adalora':
                # AdaLoRA has built-in regularization with learnable weights
                reg_loss = model.compute_regularization()
                loss_to_backward = loss + reg_loss
                train_reg_loss += reg_loss.item()
            else:
                loss_to_backward = loss
                
            accelerator.backward(loss_to_backward)
            optimizer.step()
            
            train_loss += loss.item()
            train_reg_loss += reg_loss.item()
            
        train_loss /= len(train_loader)
        train_reg_loss /= len(train_loader)
        
        # Validation
        model.eval()
        all_preds = []
        all_targets = []
        
        for batch in val_loader:
            inputs, targets = batch
            with torch.no_grad():
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                
            all_preds.extend(accelerator.gather(preds).cpu().numpy())
            all_targets.extend(accelerator.gather(targets).cpu().numpy())
            
        val_acc = accuracy_score(all_targets, all_preds)
        val_f1 = f1_score(all_targets, all_preds, average='macro')
        
        accelerator.print(f"Epoch {epoch+1}: Loss {train_loss:.4f} (Reg {train_reg_loss:.4f}) | Val Acc {val_acc:.4f} | Val F1 {val_f1:.4f}")
        
        if accelerator.is_main_process:
            if not args.no_wandb:
                wandb.log({
                    "train_loss": train_loss,
                    "reg_loss": train_reg_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "epoch": epoch
                })
                
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                best_epoch = epoch
                
                # Save checkpoint
                checkpoint_path = os.path.join(args.output_dir, f"best_model_{args.dataset}_{args.method}_shot{args.shots}_seed{args.seed}.pt")
                accelerator.save({
                    'epoch': epoch,
                    'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'args': vars(args)
                }, checkpoint_path)
                
    # Save results to CSV
    if accelerator.is_main_process:
        results = {
            "dataset": args.dataset,
            "shots": args.shots,
            "backbone": args.backbone,
            "method": args.method,
            "rank": args.rank,
            "reg_weight": args.reg_weight,
            "lr": args.lr,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "val_acc": best_val_acc,
            "val_f1": best_val_f1
        }
        df = pd.DataFrame([results])
        csv_path = os.path.join(args.output_dir, "results.csv")
        df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
        accelerator.print(f"Results saved to {csv_path}")
        if not args.no_wandb:
            wandb.finish()

if __name__ == "__main__":
    main()
