"""
Training script for CLAM model with K-Fold Cross-Validation.
Implements stratified K-fold CV for more robust evaluation on small datasets.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LambdaLR
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import pandas as pd
from copy import deepcopy

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import CLAM_SB, CLAM_MB, SmoothTop1SVM, FocalLoss, initialize_weights, initialize_attention_weights
from src.data_loader import CLAMDataset, collate_fn, RANDOM_SEED
from torch.utils.data import DataLoader


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve."""
    
    def __init__(self, patience=20, stop_epoch=30, verbose=True, delta=0.001, 
                 smoothing_factor=0.3, mode='min'):
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.delta = delta
        self.smoothing_factor = smoothing_factor
        self.mode = mode
        
        self.counter = 0
        self.best_score = None
        self.best_raw_score = None
        self.early_stop = False
        self.ema_score = None
        self.best_model_state = None
        
    def _smooth(self, score):
        if self.ema_score is None:
            self.ema_score = score
        else:
            self.ema_score = self.smoothing_factor * self.ema_score + (1 - self.smoothing_factor) * score
        return self.ema_score
    
    def _is_improvement(self, score, best):
        if self.mode == 'min':
            return score < best - self.delta
        else:
            return score > best + self.delta
        
    def __call__(self, epoch, score, model):
        smoothed_score = self._smooth(score)
        
        if self.best_score is None:
            self.best_score = smoothed_score
            self.best_raw_score = score
            self.best_model_state = deepcopy(model.state_dict())
        elif not self._is_improvement(smoothed_score, self.best_score):
            self.counter += 1
            if self.counter >= self.patience and epoch >= self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = smoothed_score
            self.best_raw_score = score
            self.best_model_state = deepcopy(model.state_dict())
            self.counter = 0
        
        return self.early_stop
    
    def get_best_model_state(self):
        return self.best_model_state


class AccuracyLogger:
    """Logger for tracking accuracy per class."""
    
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.reset()
    
    def reset(self):
        self.data = [{"count": 0, "correct": 0} for _ in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]
        if count == 0:
            return None, 0, 0
        return correct / count, correct, count


def calculate_error(Y_hat, Y):
    return 1.0 if Y_hat.item() != Y.item() else 0.0


def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, 
                    loss_fn, device, max_grad_norm=1.0, bag_dropout=0.0):
    """Training loop for CLAM with instance-level clustering."""
    model.train()
    acc_logger = AccuracyLogger(n_classes=n_classes)
    
    train_loss = 0.0
    train_error = 0.0
    train_inst_loss = 0.0
    inst_count = 0
    
    for batch_idx, batch in enumerate(loader):
        features = batch['features']
        label = batch['label']
        
        if isinstance(features, list):
            features = features[0]
            label = label[0:1]
        
        data = features.to(device)
        lbl = label.to(device)
        
        # Bag-level dropout
        if bag_dropout > 0 and data.size(0) > 10:
            n_instances = data.size(0)
            n_keep = max(10, int(n_instances * (1 - bag_dropout)))
            keep_indices = torch.randperm(n_instances)[:n_keep].sort()[0]
            data = data[keep_indices]
        
        optimizer.zero_grad()
        
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=lbl, instance_eval=True)
        
        acc_logger.log(Y_hat, lbl)
        
        bag_loss = loss_fn(logits, lbl)
        loss_value = bag_loss.item()
        
        instance_loss = instance_dict['instance_loss']
        inst_count += 1
        train_inst_loss += instance_loss.item()
        
        total_loss = bag_weight * bag_loss + (1 - bag_weight) * instance_loss
        
        train_loss += loss_value
        error = calculate_error(Y_hat, lbl)
        train_error += error
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
    
    train_loss /= len(loader)
    train_error /= len(loader)
    if inst_count > 0:
        train_inst_loss /= inst_count
    
    return train_loss, train_error, train_inst_loss


def validate_clam(model, loader, n_classes, loss_fn, device):
    """Validation loop for CLAM."""
    model.eval()
    
    val_loss = 0.0
    val_error = 0.0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            features = batch['features']
            label = batch['label']
            
            if isinstance(features, list):
                features = features[0]
                label = label[0:1]
            
            data = features.to(device)
            lbl = label.to(device)
            
            logits, Y_prob, Y_hat, _, _ = model(data, instance_eval=False)
            
            loss = loss_fn(logits, lbl)
            val_loss += loss.item()
            
            error = calculate_error(Y_hat, lbl)
            val_error += error
            
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = lbl.item()
    
    val_error /= len(loader)
    val_loss /= len(loader)
    
    # Calculate AUC
    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    return val_loss, val_error, auc, prob, labels


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


def create_model(args, instance_loss_fn, device):
    """Create and initialize a fresh model."""
    model_dict = {
        'gate': True,
        'size_arg': args.model_size,
        'dropout': args.dropout,
        'k_sample': args.k_sample,
        'n_classes': args.n_classes,
        'instance_loss_fn': instance_loss_fn,
        'subtyping': False,
        'embed_dim': args.embed_dim
    }
    
    if args.model_type == 'clam_sb':
        model = CLAM_SB(**model_dict)
    else:
        model = CLAM_MB(**model_dict)
    
    model.apply(initialize_weights)
    initialize_attention_weights(model.attention_net)
    for classifier in model.instance_classifiers:
        initialize_attention_weights(classifier)
    
    return model.to(device)


def train_fold(fold, train_patients, val_patients, df, args, device):
    """Train a single fold and return validation metrics."""
    print(f"\n{'='*60}")
    print(f"FOLD {fold + 1}/{args.n_folds}")
    print(f"{'='*60}")
    print(f"Train: {len(train_patients)} patients, Val: {len(val_patients)} patients")
    
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
    
    # Compute class weights
    train_labels = np.array([train_dataset.labels[p] for p in train_dataset.valid_patients])
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    # Initialize loss functions
    if args.use_focal_loss:
        bag_loss_fn = FocalLoss(
            alpha=class_weights, gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing
        )
    else:
        bag_loss_fn = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=args.label_smoothing
        )
    
    if args.inst_loss == 'svm':
        instance_loss_fn = SmoothTop1SVM(n_classes=2)
    else:
        instance_loss_fn = nn.CrossEntropyLoss()
    
    bag_loss_fn = bag_loss_fn.to(device)
    instance_loss_fn = instance_loss_fn.to(device)
    
    # Create fresh model for this fold
    model = create_model(args, instance_loss_fn, device)
    
    # Optimizer and schedulers
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    def warmup_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        return 1.0
    
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    
    if args.scheduler == 'cosine':
        main_scheduler = CosineAnnealingLR(
            optimizer, T_max=args.max_epochs - args.warmup_epochs, eta_min=args.min_lr
        )
    else:
        main_scheduler = None
    
    # Early stopping
    es_mode = 'min' if args.es_mode == 'loss' else 'max'
    early_stopping = EarlyStopping(
        patience=args.patience, stop_epoch=20, verbose=False,
        delta=0.001, smoothing_factor=0.3, mode=es_mode
    )
    
    # Training loop
    best_val_auc = 0.0
    fold_history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    
    for epoch in range(args.max_epochs):
        is_warmup = epoch < args.warmup_epochs
        
        # Train
        train_loss, train_error, _ = train_loop_clam(
            epoch, model, train_loader, optimizer, args.n_classes,
            args.bag_weight, bag_loss_fn, device,
            max_grad_norm=args.max_grad_norm, bag_dropout=args.bag_dropout
        )
        
        # Validate
        val_loss, val_error, val_auc, _, _ = validate_clam(
            model, val_loader, args.n_classes, bag_loss_fn, device
        )
        
        # Update schedulers
        if is_warmup:
            warmup_scheduler.step()
        elif main_scheduler is not None:
            main_scheduler.step()
        
        # Log
        fold_history['train_loss'].append(train_loss)
        fold_history['val_loss'].append(val_loss)
        fold_history['val_auc'].append(val_auc)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
        
        # Early stopping
        es_metric = val_loss if args.es_mode == 'loss' else val_auc
        if early_stopping(epoch, es_metric, model):
            print(f"  Early stopping at epoch {epoch + 1}")
            break
        
        # Print progress every 25 epochs
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_auc={val_auc:.4f}")
    
    # Load best model and get final validation predictions
    best_state = early_stopping.get_best_model_state()
    if best_state is not None:
        model.load_state_dict(best_state)
    
    val_loss, val_error, val_auc, val_probs, val_labels = validate_clam(
        model, val_loader, args.n_classes, bag_loss_fn, device
    )
    
    print(f"\n  Fold {fold + 1} Best Val AUC: {val_auc:.4f}")
    
    # Return results
    fold_results = {
        'fold': fold + 1,
        'val_auc': val_auc,
        'val_loss': val_loss,
        'val_error': val_error,
        'val_accuracy': 1 - val_error,
        'val_patients': val_patients.tolist(),
        'val_probs': val_probs.tolist(),
        'val_labels': val_labels.tolist(),
        'n_epochs': len(fold_history['train_loss']),
        'history': fold_history
    }
    
    # Save fold model
    fold_model_path = os.path.join(args.output_dir, f'model_fold{fold + 1}.pth')
    torch.save(model.state_dict(), fold_model_path)
    
    return fold_results, model


def main():
    parser = argparse.ArgumentParser(description='Train CLAM model with K-Fold Cross-Validation')
    
    # Data paths
    parser.add_argument('--clinical_csv', type=str, default='clinical_data.csv')
    parser.add_argument('--features_dir', type=str, default='features')
    parser.add_argument('--output_dir', type=str, default='results_cv')
    
    # Cross-validation
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for CV')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb'], default='clam_sb')
    parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small')
    parser.add_argument('--embed_dim', type=int, default=768, help='Feature embedding dimension')
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--k_sample', type=int, default=8)
    
    # Training parameters
    parser.add_argument('--max_epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--bag_weight', type=float, default=0.7)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--bag_dropout', type=float, default=0.15)
    
    # Loss
    parser.add_argument('--inst_loss', type=str, choices=['ce', 'svm'], default='svm')
    parser.add_argument('--use_focal_loss', action='store_true', default=False)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    
    # Early stopping
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--es_mode', type=str, choices=['loss', 'auc'], default='loss')
    
    # Scheduler
    parser.add_argument('--scheduler', type=str, choices=['cosine', 'none'], default='cosine')
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    
    # Other
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Print configuration
    print("\n" + "="*60)
    print(f"CLAM {args.n_folds}-Fold Cross-Validation")
    print("="*60)
    print(f"Model: {args.model_type} ({args.model_size})")
    print(f"Embed dim: {args.embed_dim}")
    print(f"Dropout: {args.dropout}, K-sample: {args.k_sample}")
    print(f"LR: {args.lr}, Weight decay: {args.weight_decay}")
    print(f"Bag weight: {args.bag_weight}")
    print(f"Bag dropout: {args.bag_dropout}")
    print(f"Max epochs: {args.max_epochs}")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    patients, labels, df = get_patients_and_labels(args.clinical_csv, args.features_dir)
    
    print(f"Total patients: {len(patients)}")
    print(f"Class distribution: {sum(labels)} progressors, {len(labels) - sum(labels)} responders")
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    
    # Store results from all folds
    all_fold_results = []
    all_val_probs = []
    all_val_labels = []
    all_val_patients = []
    
    # Train each fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(patients, labels)):
        train_patients = patients[train_idx]
        val_patients = patients[val_idx]
        
        fold_results, _ = train_fold(fold, train_patients, val_patients, df, args, device)
        all_fold_results.append(fold_results)
        
        # Collect predictions for overall metrics
        all_val_probs.extend(fold_results['val_probs'])
        all_val_labels.extend(fold_results['val_labels'])
        all_val_patients.extend(fold_results['val_patients'])
    
    # Compute overall metrics
    all_val_probs = np.array(all_val_probs)
    all_val_labels = np.array(all_val_labels)
    all_val_preds = np.argmax(all_val_probs, axis=1)
    
    overall_auc = roc_auc_score(all_val_labels, all_val_probs[:, 1])
    overall_acc = accuracy_score(all_val_labels, all_val_preds)
    overall_cm = confusion_matrix(all_val_labels, all_val_preds)
    
    # Per-fold metrics
    fold_aucs = [r['val_auc'] for r in all_fold_results]
    fold_accs = [r['val_accuracy'] for r in all_fold_results]
    
    # Print summary
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)
    
    print("\nPer-fold AUC:")
    for i, auc in enumerate(fold_aucs):
        print(f"  Fold {i+1}: {auc:.4f}")
    
    print(f"\nMean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"Mean Accuracy: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
    
    print(f"\nOverall AUC (all folds pooled): {overall_auc:.4f}")
    print(f"Overall Accuracy: {overall_acc:.4f}")
    
    print("\nConfusion Matrix (all folds):")
    print(overall_cm)
    
    print("\nClassification Report:")
    print(classification_report(all_val_labels, all_val_preds, 
                                target_names=['Responder', 'Progressor']))
    
    # Save results
    cv_results = {
        'n_folds': args.n_folds,
        'mean_auc': float(np.mean(fold_aucs)),
        'std_auc': float(np.std(fold_aucs)),
        'mean_accuracy': float(np.mean(fold_accs)),
        'std_accuracy': float(np.std(fold_accs)),
        'overall_auc': float(overall_auc),
        'overall_accuracy': float(overall_acc),
        'per_fold_auc': fold_aucs,
        'per_fold_accuracy': fold_accs,
        'confusion_matrix': overall_cm.tolist(),
        'fold_details': all_fold_results
    }
    
    with open(os.path.join(args.output_dir, 'cv_results.json'), 'w') as f:
        json.dump(cv_results, f, indent=2)
    
    # Save patient-level predictions
    patient_predictions = {
        'patients': all_val_patients,
        'true_labels': all_val_labels.tolist(),
        'predicted_probs': all_val_probs.tolist(),
        'predicted_labels': all_val_preds.tolist()
    }
    
    with open(os.path.join(args.output_dir, 'patient_predictions.json'), 'w') as f:
        json.dump(patient_predictions, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}/")
    print("="*60)


if __name__ == '__main__':
    main()
