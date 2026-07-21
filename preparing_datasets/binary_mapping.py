"""
Binary Solar Panel Fault Classifier - Mapping + Imbalance Handling
==================================================================
This script prepares the dataset for binary classification (Healthy vs Anomaly)
WITHOUT modifying or copying your original folder structure.

It uses:
- Strategy 1: Binary label remapping at DataLoader level
- Strategy 2: WeightedRandomSampler (oversamples Healthy class)
- Strategy 3: Weighted CrossEntropyLoss (gives higher penalty to misclassifying Healthy)

Usage:
    python binary_mapping.py --data_dir path/to/unified_Dataset_augmented
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# Original class names → Binary label (0 = Healthy, 1 = Anomaly)
BINARY_MAP = {
    "Class_0_Clean_panels": 0,      # Healthy
    "Class_1_Soiling_pollution": 1, # Anomaly
    "Class_2_Shadowing_vegetation": 1,
    "Class_3_Burn_Discoloration": 1,
    "Class_4_Structural_damage": 1,
}

CLASS_NAMES = ["Healthy", "Anomaly"]

# ──────────────────────────────────────────────
# BINARY REMAPPED DATASET (Strategy 1)
# ──────────────────────────────────────────────
class BinaryRemappedDataset(datasets.ImageFolder):
    """Custom ImageFolder that remaps 5 classes to binary (0=Healthy, 1=Anomaly)"""
    def __init__(self, root, transform=None):
        super().__init__(root=root, transform=transform)
        
        # Build remapping from original class index → binary label
        self.binary_remap = {}
        for cls_name, idx in self.class_to_idx.items():
            binary_label = BINARY_MAP.get(cls_name)
            if binary_label is None:
                raise ValueError(f"Unknown class folder: {cls_name}")
            self.binary_remap[idx] = binary_label
        
        # Update targets to binary labels
        self.targets = [self.binary_remap[label] for label in self.targets]
        self.imgs = [(path, self.binary_remap[label]) for path, label in self.imgs]

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)   # Get image with original transform
        binary_label = self.targets[index]    # Already remapped binary label
        return img, binary_label


# ──────────────────────────────────────────────
# WEIGHTED SAMPLER (Strategy 2)
# ──────────────────────────────────────────────
def get_weighted_sampler(dataset):
    """Creates WeightedRandomSampler to balance batches (oversamples minority class)"""
    # Use the binary targets we already set in BinaryRemappedDataset
    labels = dataset.targets
    
    class_counts = np.bincount(labels)
    print(f"Class distribution in dataset: Healthy={class_counts[0]}, Anomaly={class_counts[1]}")
    
    # Weight = 1 / count → minority class gets higher weight
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]
    
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler, class_counts


# ──────────────────────────────────────────────
# DATA LOADERS BUILDER (All 3 strategies combined)
# ──────────────────────────────────────────────
def build_binary_dataloaders(data_dir: str, val_split=0.15, test_split=0.15):
    """Build train/val/test loaders with all three imbalance strategies"""
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load full dataset with binary remapping
    full_dataset = BinaryRemappedDataset(data_dir, transform=None)
    
    # Stratified split: 70% train, 15% val, 15% test
    indices = list(range(len(full_dataset)))
    labels = full_dataset.targets
    
    train_idx, temp_idx = train_test_split(
        indices, test_size=(val_split + test_split), stratify=labels, random_state=SEED
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=test_split/(val_split + test_split), 
        stratify=[labels[i] for i in temp_idx], random_state=SEED
    )

    # Create subsets with proper transforms
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    test_dataset = torch.utils.data.Subset(full_dataset, test_idx)
    
    # Apply transforms (we override __getitem__ to apply transform)
    def apply_transform(dataset, transform):
        dataset.dataset.transform = transform
        return dataset
    
    train_dataset = apply_transform(train_dataset, train_transform)
    val_dataset = apply_transform(val_dataset, val_transform)
    test_dataset = apply_transform(test_dataset, val_transform)

    # Strategy 2: Weighted Sampler for training
    sampler, class_counts = get_weighted_sampler(train_dataset.dataset)
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # Strategy 3: Weighted Loss (dynamic based on actual counts)
    total_samples = len(train_dataset)
    # Higher weight for Healthy (minority class)
    loss_weights = torch.tensor([
        total_samples / (2 * class_counts[0]),   # Healthy weight
        total_samples / (2 * class_counts[1])    # Anomaly weight
    ], dtype=torch.float32)

    print(f"\nFinal splits → Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    print(f"Loss class weights: Healthy={loss_weights[0]:.3f}, Anomaly={loss_weights[1]:.3f}")

    return train_loader, val_loader, test_loader, loss_weights


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binary Mapping + Imbalance Handling for Solar Panel Dataset")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="unified_dataset_augmented")
    parser.add_argument("--val_split", type=float, default=0.15,
                        help="Validation split ratio")
    parser.add_argument("--test_split", type=float, default=0.15,
                        help="Test split ratio")
    
    args = parser.parse_args()

    print("=== Binary Solar Panel Fault Classifier - Data Preparation ===\n")
    
    train_loader, val_loader, test_loader, loss_weights = build_binary_dataloaders(
        data_dir=args.data_dir,
        val_split=args.val_split,
        test_split=args.test_split
    )

    print("\n✅ All three imbalance strategies applied successfully!")
    print("   • Strategy 1: Binary label remapping at load time")
    print("   • Strategy 2: WeightedRandomSampler (balanced batches)")
    print("   • Strategy 3: Weighted CrossEntropyLoss")
    print("\nYou can now import `build_binary_dataloaders` in your training scripts/notebooks.")