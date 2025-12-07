"""Data loading and few-shot sampling utilities."""
import os
import random
from typing import Tuple, Optional
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
from collections import defaultdict

def get_transforms(backbone_name: str) -> Tuple[transforms.Compose, transforms.Compose]:
    """Get train and validation transforms based on backbone.
    
    Args:
        backbone_name: Name of the backbone model (determines input size and normalization)
        
    Returns:
        Tuple of (train_transform, val_transform)
    """
    # Determine input size
    size = 384 if '384' in backbone_name else 224
    
    # Determine normalization
    if 'clip' in backbone_name.lower():
        mean = (0.48145466, 0.4578275, 0.40821073)  # CLIP normalization
        std = (0.26862954, 0.26130258, 0.27577711)
    else:
        mean = (0.485, 0.456, 0.406)  # ImageNet normalization
        std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    return train_transform, val_transform

def get_dataset(name: str, root: str, backbone_name: str) -> Tuple[Dataset, Dataset]:
    """Load train and validation datasets.
    
    Args:
        name: Dataset name ('stanford_cars', 'fgvc_aircraft', 'eurosat')
        root: Root directory for dataset storage
        backbone_name: Backbone model name (for transform selection)
        
    Returns:
        Tuple of (train_dataset, val_dataset)
        
    Raises:
        ValueError: If dataset name is not recognized
        RuntimeError: If dataset download fails
    """
    train_transform, val_transform = get_transforms(backbone_name)
    os.makedirs(root, exist_ok=True)
    
    if name == 'stanford_cars':
        try:
            train_set = datasets.StanfordCars(root=root, split='train', 
                                            download=True, transform=train_transform)
            val_set = datasets.StanfordCars(root=root, split='test', 
                                          download=True, transform=val_transform)
        except RuntimeError as e:
            print(f"Warning: StanfordCars download failed. Please ensure data is in {root}")
            raise e

    elif name == 'fgvc_aircraft':
        train_set = datasets.FGVCAircraft(root=root, split='trainval', 
                                         download=True, transform=train_transform)
        val_set = datasets.FGVCAircraft(root=root, split='test', 
                                       download=True, transform=val_transform)
        
    elif name == 'eurosat':
        # EuroSAT: create deterministic 80/20 split
        full_set = datasets.EuroSAT(root=root, download=True, transform=train_transform)
        val_full_set = datasets.EuroSAT(root=root, download=True, transform=val_transform)
        
        # Generate deterministic split
        g = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(full_set), generator=g).tolist()
        
        split = int(0.8 * len(full_set))
        train_indices, val_indices = indices[:split], indices[split:]
        
        train_set = Subset(full_set, train_indices)
        val_set = Subset(val_full_set, val_indices)
        
        # Attach targets for few-shot sampling
        train_set.targets = [full_set.targets[i] for i in train_indices]
        val_set.targets = [val_full_set.targets[i] for i in val_indices]

    else:
        raise ValueError(f"Unknown dataset: {name}")
        
    return train_set, val_set

class FewShotSampler:
    """Sample exactly K shots per class for few-shot learning.
    
    Args:
        dataset: PyTorch dataset with targets attribute
        k_shots: Number of samples per class
        seed: Random seed for sampling
    """
    
    def __init__(self, dataset: Dataset, k_shots: int, seed: int = 0):
        self.dataset = dataset
        self.k_shots = k_shots
        self.seed = seed
        self.indices = self._sample()

    def _sample(self) -> list:
        """Sample K shots per class.
        
        Returns:
            List of sampled indices
        """
        random.seed(self.seed)
        
        # Get targets from dataset
        targets = getattr(self.dataset, 'targets', None)
        if targets is None:
            if hasattr(self.dataset, '_labels'):
                targets = self.dataset._labels
            elif hasattr(self.dataset, 'labels'):
                targets = self.dataset.labels
            else:
                raise ValueError("Dataset does not have explicit targets/labels attribute.")
        
        # Group indices by class
        class_indices = defaultdict(list)
        for idx, label in enumerate(targets):
            class_indices[label].append(idx)
            
        # Sample K per class
        sampled_indices = []
        for label, indices in class_indices.items():
            if len(indices) < self.k_shots:
                print(f"Warning: Class {label} has fewer than {self.k_shots} samples ({len(indices)}). Taking all.")
                sampled_indices.extend(indices)
            else:
                sampled_indices.extend(random.sample(indices, self.k_shots))
                
        return sampled_indices

def get_dataloader(dataset: Dataset, batch_size: int, k_shots: Optional[int] = None, 
                   seed: int = 0, is_train: bool = True, num_workers: int = 4) -> DataLoader:
    """Create DataLoader with optional few-shot sampling.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        k_shots: Number of shots per class (None for full dataset)
        seed: Random seed for few-shot sampling
        is_train: Whether this is training data (affects shuffling)
        num_workers: Number of data loading workers
        
    Returns:
        PyTorch DataLoader
    """
    if is_train and k_shots is not None:
        sampler = FewShotSampler(dataset, k_shots, seed)
        subset = Subset(dataset, sampler.indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True, 
                          num_workers=num_workers, pin_memory=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, 
                          num_workers=num_workers, pin_memory=True)
        
    return loader
