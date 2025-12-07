"""
AdaLoRA-S: Adaptive Spectral Regularization for LoRA

Novel method that learns optimal per-layer regularization strength.
Instead of fixed λ, we learn λ_i for each layer, allowing the model
to adaptively apply regularization where beneficial.

Senior Research Scientist Implementation
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel
from peft import get_peft_model, LoraConfig
from typing import Dict


class AdaLoRASpectral(nn.Module):
    """
    Adaptive Spectral LoRA with learnable per-layer regularization.
    
    Key Innovation:
    - Learn λ_i for each LoRA layer instead of fixed global λ
    - Allows model to discover which layers benefit from orthogonality
    - No hyperparameter tuning needed (learns optimal λ values)
    
    Args:
        backbone_name: HuggingFace model identifier
        num_classes: Number of output classes
        rank: LoRA rank
        init_lambda: Initial value for learnable lambdas
        learn_lambda: Whether to make lambdas learnable
    """
    
    def __init__(self, backbone_name: str, num_classes: int, rank: int = 8,
                 init_lambda: float = 0.01, learn_lambda: bool = True):
        super().__init__()
        
        print(f"Loading AdaLoRA-S with backbone: {backbone_name}")
        
        # Load backbone
        if 'clip' in backbone_name or 'siglip' in backbone_name:
            full_model = AutoModel.from_pretrained(backbone_name)
            self.backbone = full_model.vision_model if hasattr(full_model, 'vision_model') else full_model
        else:
            self.backbone = AutoModel.from_pretrained(backbone_name)
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Determine target modules
        target_modules = ["query", "value"]
        if 'clip' in backbone_name or 'siglip' in backbone_name:
            target_modules = ["q_proj", "v_proj"]
        
        # Apply LoRA
        peft_config = LoraConfig(
            task_type=None,
            inference_mode=False,
            r=rank,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=target_modules
        )
        self.backbone = get_peft_model(self.backbone, peft_config)
        
        # Collect LoRA layers
        self.lora_layers = []
        for name, module in self.backbone.named_modules():
            if hasattr(module, 'lora_A'):
                self.lora_layers.append((name, module))
        
        num_lora_layers = len(self.lora_layers)
        print(f"Found {num_lora_layers} LoRA layers")
        
        # Learnable per-layer lambda (initialized small to avoid over-regularization)
        if learn_lambda:
            # Initialize with small positive values
            # Use log-space for better optimization
            self.log_lambdas = nn.Parameter(
                torch.ones(num_lora_layers) * torch.log(torch.tensor(init_lambda))
            )
        else:
            self.register_buffer(
                'log_lambdas',
                torch.ones(num_lora_layers) * torch.log(torch.tensor(init_lambda))
            )
        
        self.learn_lambda = learn_lambda
        
        # Get embedding dimension
        if hasattr(self.backbone.config, 'hidden_size'):
            embed_dim = self.backbone.config.hidden_size
        elif hasattr(self.backbone.config, 'vision_config'):
            embed_dim = self.backbone.config.vision_config.hidden_size
        else:
            # Fallback
            embed_dim = 768
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        self.backbone.print_trainable_parameters()
        print(f"Learnable lambdas: {learn_lambda}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone and classifier."""
        outputs = self.backbone(x)
        
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state[:, 0, :]
        
        return self.head(features)
    
    def compute_regularization(self) -> torch.Tensor:
        """
        Compute weighted spectral regularization with learned lambdas.
        
        Returns:
            Regularization loss (scalar)
        """
        total_reg = 0.0
        
        for i, (name, module) in enumerate(self.lora_layers):
            for adapter_name in module.lora_A.keys():
                A = module.lora_A[adapter_name].weight
                B = module.lora_B[adapter_name].weight
                
                r = A.shape[0]
                I = torch.eye(r, device=A.device, dtype=A.dtype)
                
                # Compute orthogonality loss
                orthog_loss = (
                    torch.norm(torch.matmul(A, A.T) - I, p='fro')**2 +
                    torch.norm(torch.matmul(B.T, B) - I, p='fro')**2
                )
                
                # Weight by learned lambda (use exp to ensure positivity)
                lambda_i = torch.exp(self.log_lambdas[i])
                total_reg += lambda_i * orthog_loss
        
        return total_reg
    
    def get_lambda_stats(self) -> Dict:
        """Return statistics about learned lambdas."""
        lambdas = torch.exp(self.log_lambdas).detach().cpu().numpy()
        
        return {
            'lambdas': lambdas.tolist(),
            'mean': float(lambdas.mean()),
            'std': float(lambdas.std()),
            'min': float(lambdas.min()),
            'max': float(lambdas.max()),
            'median': float(np.median(lambdas)),
        }
    
    def get_layer_lambdas(self) -> Dict[str, float]:
        """Return lambda value for each layer."""
        lambdas = torch.exp(self.log_lambdas).detach().cpu().numpy()
        return {name: float(lambdas[i]) for i, (name, _) in enumerate(self.lora_layers)}


def get_model(backbone_name: str, method: str, num_classes: int, 
              rank: int = 8, init_lambda: float = 0.01) -> nn.Module:
    """
    Factory function to create models.
    
    Supports:
    - 'linear': Linear probe
    - 'lora': Standard LoRA
    - 'spectral_lora': Fixed spectral regularization
    - 'orthinit_lora': Orthogonal initialization
    - 'adalora': Adaptive spectral regularization
    """
    if method == 'adalora':
        return AdaLoRASpectral(
            backbone_name,
            num_classes,
            rank=rank,
            init_lambda=init_lambda,
            learn_lambda=True
        )
    else:
        # Import from original models.py
        from models import SpectralLoRAClassifier
        return SpectralLoRAClassifier(
            backbone_name,
            num_classes,
            method=method,
            rank=rank
        )
