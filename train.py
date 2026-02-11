import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
from copy import deepcopy

# Import configuration
import config

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import CLAM_SB, CLAM_MB, SmoothTop1SVM, FocalLoss, initialize_weights, initialize_attention_weights
from src.data_loader import CLAMDataset, collate_fn
from torch.utils.data import DataLoader


def get_patients_and_labels(clinical_csv, features_dir):
    """Get all patients with features and their labels."""
    df = pd.read_csv(clinical_csv)
    id_col = df.columns[0]
    
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
    
    return np.array(all_patients), np.array(all_labels), df


def train_epoch(model, loader, optimizer, bag_weight, loss_fn, device, max_grad_norm=1.0, bag_dropout=0.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        features = batch['features']
        label = batch['label']
        
        if isinstance(features, list):
            features = features[0]
            label = label[0:1]
        
        data = features.to(device)
        lbl = label.to(device)
        
        # Bag dropout
        if bag_dropout > 0 and data.size(0) > 10:
            n_keep = max(10, int(data.size(0) * (1 - bag_dropout)))
            keep_idx = torch.randperm(data.size(0))[:n_keep].sort()[0]
            data = data[keep_idx]
        
        optimizer.zero_grad()
        logits, _, _, _, instance_dict = model(data, label=lbl, instance_eval=True)
        
        bag_loss = loss_fn(logits, lbl)
        instance_loss = instance_dict['instance_loss']
        total = bag_weight * bag_loss + (1 - bag_weight) * instance_loss
        
        total_loss += bag_loss.item()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
    
    return total_loss / len(loader)


def validate(model, loader, loss_fn, device):
    """Validate and return metrics."""
    model.eval()
    val_loss = 0.0
    probs = []
    labels = []
    
    with torch.no_grad():
        for batch in loader:
            features = batch['features']
            label = batch['label']
            
            if isinstance(features, list):
                features = features[0]
                label = label[0:1]
            
            data = features.to(device)
            lbl = label.to(device)
            
            logits, Y_prob, _, _, _ = model(data, instance_eval=False)
            val_loss += loss_fn(logits, lbl).item()
            probs.append(Y_prob.cpu().numpy())
            labels.append(lbl.item())
    
    probs = np.vstack(probs)
    labels = np.array(labels)
    
    auc = roc_auc_score(labels, probs[:, 1]) if len(np.unique(labels)) > 1 else 0.5
    acc = accuracy_score(labels, np.argmax(probs, axis=1))
    
    return val_loss / len(loader), auc, acc, probs, labels


def create_model(instance_loss_fn, device):
    """Create fresh model using variables from config."""
    model_dict = {
        'gate': config.gate,
        'size_arg': config.model_size,
        'dropout': config.dropout,
        'k_sample': config.k_sample,
        'n_classes': 2,
        'instance_loss_fn': instance_loss_fn,
        'subtyping': config.subtyping,
        'embed_dim': config.embed_dim
    }
    
    model = CLAM_SB(**model_dict) if config.model_type == 'clam_sb' else CLAM_MB(**model_dict)
    model.apply(initialize_weights)
    initialize_attention_weights(model.attention_net)
    for clf in model.instance_classifiers:
        initialize_attention_weights(clf)
    
    return model.to(device)


def main():
    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    device = torch.device('cuda' if config.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    os.makedirs(config.output_dir, exist_ok=True)
    patients, labels, df = get_patients_and_labels(config.clinical_csv, config.features_dir)
    trainval_patients, test_patients, trainval_labels, test_labels = train_test_split(
        patients, labels,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=labels
    )
    train_patients, val_patients, train_labels, val_labels = train_test_split(
        trainval_patients, trainval_labels,
        test_size=config.val_size,
        random_state=config.seed,
        stratify=trainval_labels
    )
    train_dataset = CLAMDataset(train_patients, df, config.features_dir)
    val_dataset = CLAMDataset(val_patients, df, config.features_dir)
    test_dataset = CLAMDataset(test_patients, df, config.features_dir)
    
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        num_workers=config.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=config.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=config.num_workers, collate_fn=collate_fn, pin_memory=True
    )

    # Class weights for imbalanced loss
    train_labels_array = np.array([train_dataset.labels[p] for p in train_dataset.valid_patients])
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels_array), y=train_labels_array)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    # Loss functions
    if config.use_focal_loss:
        bag_loss_fn = FocalLoss(alpha=class_weights, gamma=config.focal_gamma, 
                                label_smoothing=config.label_smoothing).to(device)
    else:
        bag_loss_fn = nn.CrossEntropyLoss(weight=class_weights, 
                                           label_smoothing=config.label_smoothing).to(device)
    
    instance_loss_fn = SmoothTop1SVM(n_classes=2).to(device)
    
    # Initialize Model
    model = create_model(instance_loss_fn, device)
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # Schedulers
    def warmup_lambda(epoch):
        if epoch < config.warmup_epochs:
            return (epoch + 1) / config.warmup_epochs
        return 1.0
    
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=config.max_epochs - config.warmup_epochs, 
                                       eta_min=config.min_lr)

    best_val_auc = 0.0
    best_epoch = 0
    
    print(f"Starting training for {config.max_epochs} epochs...")
    
    for epoch in range(config.max_epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, config.bag_weight, 
            bag_loss_fn, device, max_grad_norm=config.max_grad_norm, 
            bag_dropout=config.bag_dropout
        )
        
        val_loss, val_auc, val_acc, _, _ = validate(model, val_loader, bag_loss_fn, device)
        
        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            model_path = os.path.join(config.output_dir, 'best_model.pt')
            torch.save(model.state_dict(), model_path)
            print(f"  >>> New Best Model Saved (AUC: {val_auc:.4f})")
        
        if epoch < config.warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()

    print(f"Training finished. Best AUC {best_val_auc:.4f} at epoch {best_epoch+1}.")

if __name__ == '__main__':
    main()