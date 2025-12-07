"""Model definitions and spectral regularization for LoRA."""
from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
import math


def orthogonal_init_lora_weights(module, rank: int, scaling: float = 1.0):
    """
    Initialize LoRA weights with orthogonal structure.
    
    This initialization strategy creates LoRA matrices that start in an
    orthogonal subspace, potentially providing benefits without regularization.
    
    Args:
        module: PEFT LoRA module containing lora_A and lora_B
        rank: LoRA rank
        scaling: Initialization scaling factor
    """
    for adapter_name in module.lora_A.keys():
        # Get dimensions
        A_weight = module.lora_A[adapter_name].weight  # (rank, in_features)
        B_weight = module.lora_B[adapter_name].weight  # (out_features, rank)
        
        r, in_features = A_weight.shape
        out_features, _ = B_weight.shape
        
        # Initialize A with orthogonal rows using QR decomposition
        A_init = torch.randn(in_features, r)
        Q_A, R_A = torch.linalg.qr(A_init)
        A_orth = Q_A.T * scaling  # (r, in_features)
        
        # Initialize B with orthogonal columns using QR decomposition
        B_init = torch.randn(out_features, r)
        Q_B, R_B = torch.linalg.qr(B_init)
        B_orth = Q_B * scaling  # (out_features, r)
        
        # Apply orthogonal initialization
        module.lora_A[adapter_name].weight.data = A_orth
        module.lora_B[adapter_name].weight.data = B_orth


class SpectralLoRAClassifier(nn.Module):
    """Classifier with optional LoRA or Spectral-LoRA adaptation.
    
    This model wraps vision transformers and adds a classification head.
    It supports three modes:
    - 'linear': Only train the classification head
    - 'lora': Standard LoRA adaptation
    - 'spectral_lora': LoRA with orthogonality-enforcing regularization
    
    Args:
        backbone_name: HuggingFace model identifier
        num_classes: Number of output classes
        method: Adaptation method ('linear', 'lora', or 'spectral_lora')
        rank: LoRA rank (default: 8)
        alpha: LoRA alpha parameter (default: 16)
    """
    
    def __init__(self, backbone_name: str, num_classes: int, method: str = 'lora', 
                 rank: int = 8, alpha: int = 16):
        super().__init__()
        self.method = method
        
        # Load Backbone
        # We use AutoModel to get the base transformer.
        # For CLIP/SigLIP, we might need to be careful to get just the vision tower if possible,
        # or just ignore the text part.
        # 'openai/clip-vit-base-patch32' -> CLIPModel. Vision model is .vision_model
        # 'google/siglip-...' -> SiglipModel. Vision model is .vision_model
        # 'facebook/dinov2-...' -> Dinov2Model.
        
        print(f"Loading backbone: {backbone_name}")
        if 'clip' in backbone_name or 'siglip' in backbone_name:
            full_model = AutoModel.from_pretrained(backbone_name)
            if hasattr(full_model, 'vision_model'):
                self.backbone = full_model.vision_model
            else:
                self.backbone = full_model # Fallback
        else:
            self.backbone = AutoModel.from_pretrained(backbone_name)
            
        # Freeze backbone initially
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Determine embedding dimension
        # Try to infer from config
        if hasattr(self.backbone.config, 'hidden_size'):
            self.embed_dim = self.backbone.config.hidden_size
        elif hasattr(self.backbone.config, 'vision_config'):
             self.embed_dim = self.backbone.config.vision_config.hidden_size
        else:
            # Dummy forward pass
            dummy_input = torch.zeros(1, 3, 224, 224) # Assuming 224
            if '384' in backbone_name:
                 dummy_input = torch.zeros(1, 3, 384, 384)
            with torch.no_grad():
                out = self.backbone(dummy_input)
                if hasattr(out, 'pooler_output') and out.pooler_output is not None:
                    self.embed_dim = out.pooler_output.shape[-1]
                else:
                    # Use last hidden state mean or CLS
                    self.embed_dim = out.last_hidden_state.shape[-1]

        # Classifier Head
        self.head = nn.Linear(self.embed_dim, num_classes)
        
        # Apply LoRA if needed
        if 'lora' in method:
            target_modules = ["q_proj", "v_proj"] # Common for ViT
            if 'dino' in backbone_name:
                 target_modules = ["query", "value"] # DINO usually uses these names? Need to verify.
                 # Actually DINOv2 uses standard ViT names usually.
                 # Let's check DINOv2 config names if possible or use a safe list.
                 # Transformers DINOv2: query, key, value, dense (in attention)
                 target_modules = ["query", "value"]
            
            # For CLIP/SigLIP (Transformers implementation)
            # CLIP: q_proj, v_proj
            # SigLIP: q_proj, v_proj
            if 'clip' in backbone_name or 'siglip' in backbone_name:
                target_modules = ["q_proj", "v_proj"]

            peft_config = LoraConfig(
                task_type=None, # We are doing custom classification
                inference_mode=False,
                r=rank,
                lora_alpha=alpha,
                lora_dropout=0.1,
                target_modules=target_modules,
                init_lora_weights=True  # Use default init first
            )
            self.backbone = get_peft_model(self.backbone, peft_config)
            
            # Apply orthogonal initialization if orthinit method
            if method == 'orthinit_lora':
                print("Applying orthogonal initialization...")
                for name, module in self.backbone.named_modules():
                    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                        orthogonal_init_lora_weights(module, rank, scaling=1.0)
                print("Orthogonal initialization complete")
            
            self.backbone.print_trainable_parameters()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Class logits [B, num_classes]
        """
        outputs = self.backbone(x)
        
        # Extract features
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            # CLS token (index 0)
            features = outputs.last_hidden_state[:, 0, :]
            
        logits = self.head(features)
        return logits

def spectral_regularization_loss(model: nn.Module) -> torch.Tensor:
    """Compute spectral regularization loss for orthogonality.
    
    Enforces orthogonality on LoRA matrices A and B:
    L_reg = sum ||(A @ A.T) - I||_F^2 + ||(B.T @ B) - I||_F^2
    
    where:
    - A is the down-projection matrix (r x d_in)
    - B is the up-projection matrix (d_out x r)
    - I is the identity matrix (r x r)
    - ||.||_F is the Frobenius norm
    
    Args:
        model: Model with LoRA layers (from PEFT)
        
    Returns:
        Regularization loss (scalar tensor)
    """
    loss = 0.0
    
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            for adapter_name in module.lora_A.keys():
                # Get LoRA matrices
                # A: (r, in_features), B: (out_features, r)
                A = module.lora_A[adapter_name].weight
                B = module.lora_B[adapter_name].weight
                
                r = A.shape[0]
                I = torch.eye(r, device=A.device, dtype=A.dtype)
                
                # Compute orthogonality loss
                # A @ A.T should be identity (r x r)
                # B.T @ B should be identity (r x r)
                term1 = torch.norm(torch.matmul(A, A.T) - I, p='fro')**2
                term2 = torch.norm(torch.matmul(B.T, B) - I, p='fro')**2
                
                loss += (term1 + term2)
                
    return loss

def get_model(backbone_name: str, method: str, num_classes: int, rank: int = 8) -> SpectralLoRAClassifier:
    """Factory function to create a model.
    
    Args:
        backbone_name: HuggingFace model identifier
        method: Adaptation method ('linear', 'lora', 'spectral_lora')
        num_classes: Number of output classes
        rank: LoRA rank (default: 8)
        
    Returns:
        Initialized SpectralLoRAClassifier
    """
    return SpectralLoRAClassifier(backbone_name, num_classes, method, rank)
