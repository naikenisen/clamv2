"""
Data loader for CLAM training pipeline.
Handles stratified train/val/test split, .pt feature loading, and variable bag sizes.
Saves splits to splits.json for reproducibility.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42
SPLITS_FILE = 'splits.json'


class CLAMDataset(Dataset):
    """
    Dataset for CLAM training.
    Loads pre-extracted features from .pt files.
    Each sample is a bag (all tiles from one patient).
    """
    
    def __init__(self, patient_ids, labels_df, features_dir, label_col='status'):
        """
        Args:
            patient_ids: List of patient IDs to include
            labels_df: DataFrame with patient_id and label columns
            features_dir: Directory containing .pt feature files
            label_col: Name of the label column in labels_df
        """
        self.patient_ids = patient_ids
        self.features_dir = features_dir
        self.label_col = label_col
        
        # Create patient_id to label mapping
        # Handle the column name issue (patient+AF8-id -> patient_id)
        id_col = labels_df.columns[0]  # First column is patient_id
        self.labels = {}
        for _, row in labels_df.iterrows():
            pid = str(int(row[id_col])) if pd.notna(row[id_col]) else None
            if pid and pd.notna(row[label_col]):
                self.labels[pid] = int(row[label_col])
        
        # Filter patient_ids to only those with labels and features
        self.valid_patients = []
        for pid in patient_ids:
            pid_str = str(pid)
            feature_path = os.path.join(features_dir, f"{pid_str}.pt")
            if pid_str in self.labels and os.path.exists(feature_path):
                self.valid_patients.append(pid_str)
        
        print(f"Dataset initialized with {len(self.valid_patients)} patients")
    
    def __len__(self):
        return len(self.valid_patients)
    
    def __getitem__(self, idx):
        patient_id = self.valid_patients[idx]
        
        # Load features
        feature_path = os.path.join(self.features_dir, f"{patient_id}.pt")
        data = torch.load(feature_path, weights_only=False)
        
        # Handle both new format (dict) and legacy format (tensor only)
        if isinstance(data, dict):
            features = data['features']  # Shape: (N, feature_dim)
            coords = data['coords']      # Shape: (N, 2) in (x, y) pixel format
            tile_names = data.get('tile_names', [])
        else:
            # Legacy format: data is just the features tensor
            features = data
            # Create dummy coordinates (will break heatmap generation)
            coords = torch.zeros((features.shape[0], 2), dtype=torch.long)
            tile_names = []
            print(f"  Warning: {patient_id} uses legacy format without coordinates")
        
        # Get label
        label = self.labels[patient_id]
        
        return {
            'features': features,
            'coords': coords,
            'label': torch.tensor(label, dtype=torch.long),
            'patient_id': patient_id,
            'tile_names': tile_names
        }


def collate_fn(batch):
    """
    Custom collate function for variable bag sizes.
    Since bag sizes vary, we process one bag at a time (batch_size=1).
    For batch_size > 1, we would need padding.
    """
    if len(batch) == 1:
        # Single sample - no need for padding
        item = batch[0]
        return {
            'features': item['features'],      # (N, feature_dim)
            'coords': item['coords'],          # (N, 2)
            'label': item['label'].unsqueeze(0),  # (1,)
            'patient_id': [item['patient_id']],
            'tile_names': [item['tile_names']]
        }
    else:
        # Multiple samples - would need padding
        # For simplicity, we concatenate but keep track of bag boundaries
        # This is mainly for batch_size=1 use case
        features_list = [item['features'] for item in batch]
        coords_list = [item['coords'] for item in batch]
        labels = torch.stack([item['label'] for item in batch])
        patient_ids = [item['patient_id'] for item in batch]
        tile_names = [item['tile_names'] for item in batch]
        
        # For batch > 1, return lists (model should handle one at a time)
        return {
            'features': features_list,
            'coords': coords_list,
            'label': labels,
            'patient_id': patient_ids,
            'tile_names': tile_names
        }


def create_splits(clinical_csv, features_dir, test_size=0.15, val_size=0.15, 
                  random_seed=RANDOM_SEED, save_path=SPLITS_FILE):
    """
    Create stratified train/val/test splits.
    
    Args:
        clinical_csv: Path to clinical data CSV
        features_dir: Directory with .pt feature files
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining after test)
        random_seed: Random seed for reproducibility
        save_path: Path to save splits JSON
        
    Returns:
        Dictionary with train, val, test patient ID lists
    """
    # Load clinical data
    df = pd.read_csv(clinical_csv)
    id_col = df.columns[0]  # First column is patient_id
    
    # Get patients with both features and labels
    all_patients = []
    all_labels = []
    
    for _, row in df.iterrows():
        if pd.isna(row[id_col]) or pd.isna(row['status']):
            continue
        pid = str(int(row[id_col]))
        feature_path = os.path.join(features_dir, f"{pid}.pt")
        if os.path.exists(feature_path):
            all_patients.append(pid)
            all_labels.append(int(row['status']))
    
    print(f"Total patients with features and labels: {len(all_patients)}")
    print(f"Class distribution: {sum(all_labels)} progressors, {len(all_labels) - sum(all_labels)} responders")
    
    # Stratified split: first split off test set
    patients_trainval, patients_test, labels_trainval, labels_test = train_test_split(
        all_patients, all_labels, 
        test_size=test_size, 
        random_state=random_seed, 
        stratify=all_labels
    )
    
    # Split trainval into train and val
    val_ratio = val_size / (1 - test_size)  # Adjust ratio for remaining data
    patients_train, patients_val, labels_train, labels_val = train_test_split(
        patients_trainval, labels_trainval,
        test_size=val_ratio,
        random_state=random_seed,
        stratify=labels_trainval
    )
    
    splits = {
        'train': patients_train,
        'val': patients_val,
        'test': patients_test
    }
    
    # Save splits
    with open(save_path, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\nSplits saved to {save_path}")
    print(f"Train: {len(patients_train)} patients")
    print(f"Val: {len(patients_val)} patients")
    print(f"Test: {len(patients_test)} patients")
    
    return splits


def load_splits(splits_path=SPLITS_FILE):
    """
    Load pre-existing splits from JSON file.
    """
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"Splits file not found: {splits_path}")
    
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    
    return splits


def get_dataloaders(clinical_csv, features_dir, batch_size=1, 
                    num_workers=4, create_new_splits=True,
                    test_size=0.15, val_size=0.15,
                    random_seed=RANDOM_SEED, splits_path=SPLITS_FILE):
    """
    Get train, validation, and test dataloaders.
    
    Args:
        clinical_csv: Path to clinical data CSV
        features_dir: Directory with .pt feature files
        batch_size: Batch size (recommended: 1 for variable bag sizes)
        num_workers: Number of data loading workers
        create_new_splits: Whether to create new splits or load existing
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_seed: Random seed
        splits_path: Path to splits JSON file
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Load clinical data
    df = pd.read_csv(clinical_csv)
    
    # Get or create splits
    if create_new_splits or not os.path.exists(splits_path):
        splits = create_splits(
            clinical_csv, features_dir, 
            test_size=test_size, val_size=val_size,
            random_seed=random_seed, save_path=splits_path
        )
    else:
        splits = load_splits(splits_path)
        print(f"Loaded existing splits from {splits_path}")
        print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    
    # Create datasets
    train_dataset = CLAMDataset(splits['train'], df, features_dir)
    val_dataset = CLAMDataset(splits['val'], df, features_dir)
    test_dataset = CLAMDataset(splits['test'], df, features_dir)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_test_loader_from_splits(clinical_csv, features_dir, splits_path=SPLITS_FILE,
                                 batch_size=1, num_workers=4):
    """
    Get only the test dataloader using pre-existing splits.
    Used by infer.py to ensure the same test set is used.
    
    Args:
        clinical_csv: Path to clinical data CSV
        features_dir: Directory with .pt feature files
        splits_path: Path to splits JSON file
        batch_size: Batch size
        num_workers: Number of workers
        
    Returns:
        test_loader, test_patient_ids
    """
    # Load splits
    splits = load_splits(splits_path)
    df = pd.read_csv(clinical_csv)
    
    # Create test dataset
    test_dataset = CLAMDataset(splits['test'], df, features_dir)
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return test_loader, splits['test']


if __name__ == '__main__':
    # Test the data loading
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--clinical_csv', type=str, default='clinical_data.csv')
    parser.add_argument('--features_dir', type=str, default='features')
    args = parser.parse_args()
    
    # Test creating splits and dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        args.clinical_csv,
        args.features_dir,
        batch_size=1,
        num_workers=0,
        create_new_splits=True
    )
    
    # Test loading a batch
    for batch in train_loader:
        print(f"Features shape: {batch['features'].shape}")
        print(f"Label: {batch['label']}")
        print(f"Patient ID: {batch['patient_id']}")
        break
