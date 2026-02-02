"""
Grid Search with Cross-Validation for CLAM hyperparameters.
Tests multiple bag_weight values and other hyperparameters to find optimal configuration.
After finding best config, trains final model and saves weights.

For inference and attention map generation, use infer.py
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm
import pandas as pd
from copy import deepcopy
from itertools import product
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import CLAM_SB, CLAM_MB, SmoothTop1SVM, FocalLoss, initialize_weights, initialize_attention_weights
from src.data_loader import CLAMDataset, collate_fn
from torch.utils.data import DataLoader


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve."""
    
    def __init__(self, patience=15, stop_epoch=20, mode='min'):
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.best_model_state = None
        self.early_stop = False
        
    def __call__(self, epoch, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = deepcopy(model.state_dict())
        else:
            is_better = (score < self.best_score) if self.mode == 'min' else (score > self.best_score)
            if is_better:
                self.best_score = score
                self.best_model_state = deepcopy(model.state_dict())
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience and epoch >= self.stop_epoch:
                    self.early_stop = True
        return self.early_stop


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


def validate(model, loader, n_classes, loss_fn, device):
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


def create_model(args, instance_loss_fn, device):
    """Create fresh model."""
    model_dict = {
        'gate': True,
        'size_arg': args.model_size,
        'dropout': args.dropout,
        'k_sample': args.k_sample,
        'n_classes': 2,
        'instance_loss_fn': instance_loss_fn,
        'subtyping': False,
        'embed_dim': args.embed_dim
    }
    
    model = CLAM_SB(**model_dict) if args.model_type == 'clam_sb' else CLAM_MB(**model_dict)
    model.apply(initialize_weights)
    initialize_attention_weights(model.attention_net)
    for clf in model.instance_classifiers:
        initialize_attention_weights(clf)
    
    return model.to(device)


def run_cv_for_config(config, patients, labels, df, args, device, n_folds=5):
    """
    Run cross-validation for a specific hyperparameter configuration.
    Returns mean AUC and std.
    """
    bag_weight = config['bag_weight']
    dropout = config.get('dropout', args.dropout)
    k_sample = config.get('k_sample', args.k_sample)
    lr = config.get('lr', args.lr)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
    fold_aucs = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(patients, labels)):
        train_patients = patients[train_idx]
        val_patients = patients[val_idx]
        
        # Create datasets
        train_dataset = CLAMDataset(train_patients, df, args.features_dir)
        val_dataset = CLAMDataset(val_patients, df, args.features_dir)
        
        train_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=True,
            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
        )
        
        # Class weights
        train_labels = np.array([train_dataset.labels[p] for p in train_dataset.valid_patients])
        class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        
        # Loss functions
        if args.use_focal_loss:
            bag_loss_fn = FocalLoss(alpha=class_weights, gamma=args.focal_gamma, 
                                    label_smoothing=args.label_smoothing).to(device)
        else:
            bag_loss_fn = nn.CrossEntropyLoss(weight=class_weights, 
                                               label_smoothing=args.label_smoothing).to(device)
        
        instance_loss_fn = SmoothTop1SVM(n_classes=2).to(device)
        
        # Create model with current config
        args_copy = argparse.Namespace(**vars(args))
        args_copy.dropout = dropout
        args_copy.k_sample = k_sample
        model = create_model(args_copy, instance_loss_fn, device)
        
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        
        # Warmup + cosine scheduler
        def warmup_lambda(epoch):
            if epoch < args.warmup_epochs:
                return (epoch + 1) / args.warmup_epochs
            return 1.0
        
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
        main_scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs - args.warmup_epochs, 
                                           eta_min=args.min_lr)
        
        early_stopping = EarlyStopping(patience=args.patience, stop_epoch=15, mode='min')
        
        # Training loop
        best_auc = 0.0
        for epoch in range(args.max_epochs):
            train_epoch(model, train_loader, optimizer, bag_weight, bag_loss_fn, device,
                       max_grad_norm=args.max_grad_norm, bag_dropout=args.bag_dropout)
            
            val_loss, val_auc, _, _, _ = validate(model, val_loader, 2, bag_loss_fn, device)
            
            if epoch < args.warmup_epochs:
                warmup_scheduler.step()
            else:
                main_scheduler.step()
            
            if val_auc > best_auc:
                best_auc = val_auc
            
            if early_stopping(epoch, val_loss, model):
                break
        
        # À la fin de run_cv_for_config, restaurer et retourner le meilleur modèle
        model.load_state_dict(early_stopping.best_model_state)
        
        fold_aucs.append(best_auc)
    
    return np.mean(fold_aucs), np.std(fold_aucs), fold_aucs


def main():
    parser = argparse.ArgumentParser(description='Grid Search for CLAM hyperparameters')
    
    # Data paths
    parser.add_argument('--clinical_csv', type=str, default='clinical_data.csv')
    parser.add_argument('--features_dir', type=str, default='features')
    # Default output directory uses current date
    default_output_dir = f"results_{datetime.now().strftime('%Y-%m-%d')}"
    parser.add_argument('--output_dir', type=str, default=default_output_dir)
    
    # Grid search parameters
    parser.add_argument('--bag_weights', type=str, default='0.5,0.6,0.7,0.8,0.9',
                        help='Comma-separated bag_weight values to test')
    parser.add_argument('--dropouts', type=str, default='0.5',
                        help='Comma-separated dropout values to test')
    parser.add_argument('--k_samples', type=str, default='8',
                        help='Comma-separated k_sample values to test')
    parser.add_argument('--lrs', type=str, default='0.0001',
                        help='Comma-separated learning rate values to test')
    
    # CV settings
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    
    # Model parameters (fixed)
    parser.add_argument('--model_type', type=str, default='clam_sb')
    parser.add_argument('--model_size', type=str, default='small')
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--k_sample', type=int, default=8)
    
    # Training parameters (fixed)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--bag_dropout', type=float, default=0.15)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    
    # Loss
    parser.add_argument('--use_focal_loss', action='store_true', default=False)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    
    # Other
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test_size', type=float, default=0.15,
                        help='Proportion of data to use for test set')
    
    args = parser.parse_args()
    
    # Parse grid search values
    bag_weights = [float(x) for x in args.bag_weights.split(',')]
    dropouts = [float(x) for x in args.dropouts.split(',')]
    k_samples = [int(x) for x in args.k_samples.split(',')]
    lrs = [float(x) for x in args.lrs.split(',')]
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    patients, labels, df = get_patients_and_labels(args.clinical_csv, args.features_dir)
    print(f"Total patients: {len(patients)}")
    print(f"Class distribution: {sum(labels)} progressors, {len(labels) - sum(labels)} responders")
    
    # Split off test set first (stratified)
    print(f"\nSplitting data: {int((1-args.test_size)*100)}% train+val, {int(args.test_size*100)}% test")
    trainval_patients, test_patients, trainval_labels, test_labels = train_test_split(
        patients, labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels
    )
    print(f"Train+Val: {len(trainval_patients)} patients")
    print(f"Test: {len(test_patients)} patients")
    print(f"Test class distribution: {sum(test_labels)} progressors, {len(test_labels) - sum(test_labels)} responders")
    
    # Generate all configurations
    configs = []
    for bw, do, ks, lr in product(bag_weights, dropouts, k_samples, lrs):
        configs.append({
            'bag_weight': bw,
            'dropout': do,
            'k_sample': ks,
            'lr': lr
        })
    
    print(f"\n{'='*60}")
    print(f"GRID SEARCH: {len(configs)} configurations x {args.n_folds} folds")
    print(f"{'='*60}")
    print(f"bag_weight: {bag_weights}")
    print(f"dropout: {dropouts}")
    print(f"k_sample: {k_samples}")
    print(f"lr: {lrs}")
    print(f"{'='*60}\n")
    
    # Run grid search on train+val only
    results = []
    best_auc = 0.0
    best_config = None
    
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Testing: bag_weight={config['bag_weight']}, "
              f"dropout={config['dropout']}, k_sample={config['k_sample']}, lr={config['lr']}")
        
        mean_auc, std_auc, fold_aucs = run_cv_for_config(
            config, trainval_patients, trainval_labels, df, args, device, args.n_folds
        )
        
        result = {
            **config,
            'mean_auc': float(mean_auc),
            'std_auc': float(std_auc),
            'fold_aucs': [float(x) for x in fold_aucs]
        }
        results.append(result)
        
        print(f"  -> Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_config = config.copy()
            best_config['mean_auc'] = mean_auc
            best_config['std_auc'] = std_auc
            print(f"  *** New best! ***")
    
    # Sort results by mean AUC
    results_sorted = sorted(results, key=lambda x: x['mean_auc'], reverse=True)
    
    # Print summary
    print(f"\n{'='*60}")
    print("GRID SEARCH RESULTS (sorted by AUC)")
    print(f"{'='*60}")
    
    print(f"\n{'Rank':<5} {'bag_weight':<12} {'dropout':<10} {'k_sample':<10} {'lr':<12} {'AUC':<20}")
    print("-" * 70)
    for i, r in enumerate(results_sorted[:10]):
        auc_str = f"{r['mean_auc']:.4f} ± {r['std_auc']:.4f}"
        print(f"{i+1:<5} {r['bag_weight']:<12} {r['dropout']:<10} {r['k_sample']:<10} {r['lr']:<12} {auc_str:<20}")
    
    print(f"\n{'='*60}")
    print("BEST CONFIGURATION")
    print(f"{'='*60}")
    print(f"bag_weight: {best_config['bag_weight']}")
    print(f"dropout: {best_config['dropout']}")
    print(f"k_sample: {best_config['k_sample']}")
    print(f"lr: {best_config['lr']}")
    print(f"Mean AUC: {best_config['mean_auc']:.4f} ± {best_config['std_auc']:.4f}")
    print(f"{'='*60}")
    
    # =========================================================================
    # TRAIN FINAL MODEL with best config on all train+val data
    # =========================================================================
    print(f"\n{'='*60}")
    print("TRAINING FINAL MODEL")
    print(f"{'='*60}")
    print(f"Training on {len(trainval_patients)} patients with best hyperparameters...")
    
    # Create train+val dataset
    trainval_dataset = CLAMDataset(trainval_patients, df, args.features_dir)
    trainval_loader = DataLoader(
        trainval_dataset, batch_size=1, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    
    # Create test dataset
    test_dataset = CLAMDataset(test_patients, df, args.features_dir)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    
    # Class weights for final training
    train_labels_final = np.array([trainval_dataset.labels[p] for p in trainval_dataset.valid_patients])
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels_final), y=train_labels_final)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    # Loss functions
    if args.use_focal_loss:
        bag_loss_fn = FocalLoss(alpha=class_weights, gamma=args.focal_gamma, 
                                label_smoothing=args.label_smoothing).to(device)
    else:
        bag_loss_fn = nn.CrossEntropyLoss(weight=class_weights, 
                                           label_smoothing=args.label_smoothing).to(device)
    
    instance_loss_fn = SmoothTop1SVM(n_classes=2).to(device)
    
    # Create model with best config
    args_final = argparse.Namespace(**vars(args))
    args_final.dropout = best_config['dropout']
    args_final.k_sample = best_config['k_sample']
    final_model = create_model(args_final, instance_loss_fn, device)
    
    optimizer = Adam(final_model.parameters(), lr=best_config['lr'], weight_decay=args.weight_decay)
    
    # Warmup + cosine scheduler
    def warmup_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        return 1.0
    
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs - args.warmup_epochs, 
                                       eta_min=args.min_lr)
    
    # Training loop for final model
    best_model_state = None
    best_train_loss = float('inf')
    
    for epoch in tqdm(range(args.max_epochs), desc="Training final model"):
        train_loss = train_epoch(final_model, trainval_loader, optimizer, best_config['bag_weight'], 
                                 bag_loss_fn, device, max_grad_norm=args.max_grad_norm, 
                                 bag_dropout=args.bag_dropout)
        
        if epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()
        
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_model_state = deepcopy(final_model.state_dict())
    
    # Load best model state
    final_model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_auc, test_acc, test_probs, test_true_labels = validate(
        final_model, test_loader, 2, bag_loss_fn, device
    )
    
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Confusion matrix
    test_preds = np.argmax(test_probs, axis=1)
    cm = confusion_matrix(test_true_labels, test_preds)
    print(f"Confusion Matrix:\n{cm}")
    
    # Save model
    model_path = os.path.join(args.output_dir, 'best_model.pt')
    torch.save(final_model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'n_folds': args.n_folds,
        'n_patients_total': len(patients),
        'n_patients_trainval': len(trainval_patients),
        'n_patients_test': len(test_patients),
        'test_size': args.test_size,
        'grid_search_params': {
            'bag_weights': bag_weights,
            'dropouts': dropouts,
            'k_samples': k_samples,
            'lrs': lrs
        },
        'best_config': best_config,
        'test_metrics': {
            'auc': float(test_auc),
            'accuracy': float(test_acc),
            'confusion_matrix': cm.tolist()
        },
        'all_gridsearch_results': results_sorted
    }
    
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(output, f, indent=2)
    
    # Save best config for easy reuse
    with open(os.path.join(args.output_dir, 'best_config.json'), 'w') as f:
        json.dump(best_config, f, indent=2)
    
    # Save splits for reproducibility
    splits = {
        'trainval': trainval_patients.tolist(),
        'test': test_patients.tolist()
    }
    with open(os.path.join(args.output_dir, 'splits.json'), 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {args.output_dir}/")
    print(f"  - best_model.pt: Model weights")
    print(f"  - results.json: Full results and metrics")
    print(f"  - best_config.json: Best hyperparameters")
    print(f"  - splits.json: Train/test split")
    print(f"\nTo generate attention maps and ROC curves, run:")
    print(f"  python infer.py --results_dir {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
