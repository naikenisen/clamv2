"""
Training script for CLAM model.
Implements the dual-loss training loop with bag-level and instance-level clustering losses.
Saves the best model based on validation AUC.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import CLAM_SB, CLAM_MB, SmoothTop1SVM, initialize_weights
from src.data_loader import get_dataloaders


class EarlyStopping:
    """Early stops the training if validation AUC doesn't improve after a given patience."""
    
    def __init__(self, patience=20, stop_epoch=50, verbose=True):
        """
        Args:
            patience: How long to wait after last improvement
            stop_epoch: Minimum epochs before early stopping kicks in
            verbose: Print messages
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, epoch, val_auc, model, ckpt_path='model.pth'):
        score = val_auc
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_path)
        elif score <= self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience and epoch >= self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, ckpt_path)
            self.counter = 0
        
        return self.early_stop
    
    def save_checkpoint(self, model, ckpt_path):
        """Save model when validation AUC improves."""
        torch.save(model.state_dict(), ckpt_path)
        if self.verbose:
            print(f'Saved best model with AUC: {self.best_score:.4f}')


class AccuracyLogger:
    """Logger for tracking accuracy per class."""
    
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.correct = {i: 0 for i in range(n_classes)}
        self.count = {i: 0 for i in range(n_classes)}
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat.item())
        Y = int(Y.item())
        self.count[Y] += 1
        if Y_hat == Y:
            self.correct[Y] += 1
    
    def get_summary(self, c):
        count = self.count[c]
        correct = self.correct[c]
        if count == 0:
            return None, 0, 0
        acc = correct / count
        return acc, correct, count


def calculate_error(Y_hat, Y):
    """Calculate prediction error."""
    return 1.0 if Y_hat.item() != Y.item() else 0.0


def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, 
                    loss_fn, device, verbose=True):
    """
    Training loop for CLAM with instance-level clustering.
    
    Args:
        epoch: Current epoch number
        model: CLAM model
        loader: Training data loader
        optimizer: Optimizer
        n_classes: Number of classes
        bag_weight: Weight for bag-level loss (instance weight = 1 - bag_weight)
        loss_fn: Bag-level loss function
        device: Training device
        verbose: Whether to print progress
        
    Returns:
        train_loss, train_error, train_inst_loss
    """
    model.train()
    acc_logger = AccuracyLogger(n_classes=n_classes)
    inst_logger = AccuracyLogger(n_classes=2)  # Instance classification is binary
    
    train_loss = 0.0
    train_error = 0.0
    train_inst_loss = 0.0
    inst_count = 0
    
    for batch_idx, batch in enumerate(loader):
        # Get data
        features = batch['features']
        label = batch['label']
        
        # Handle batch_size > 1 case (features is a list)
        if isinstance(features, list):
            # Process first sample only (batch_size=1 is recommended for CLAM)
            features = features[0]
            label = label[0:1]
        
        data = features.to(device)
        lbl = label.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with instance evaluation
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=lbl, instance_eval=True)
        
        acc_logger.log(Y_hat, lbl)
        
        # Bag-level loss
        bag_loss = loss_fn(logits, lbl)
        loss_value = bag_loss.item()
        
        # Instance-level loss
        instance_loss = instance_dict['instance_loss']
        inst_count += 1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        # Combined loss: bag_weight * bag_loss + (1 - bag_weight) * instance_loss
        total_loss = bag_weight * bag_loss + (1 - bag_weight) * instance_loss
        
        # Log instance predictions
        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        for p, t in zip(inst_preds, inst_labels):
            inst_logger.log(torch.tensor([p]), torch.tensor([t]))
        
        train_loss += loss_value
        
        if verbose and (batch_idx + 1) % 20 == 0:
            print(f'  Batch {batch_idx+1}, loss: {loss_value:.4f}, '
                  f'instance_loss: {instance_loss_value:.4f}, '
                  f'total_loss: {total_loss.item():.4f}, '
                  f'label: {lbl.item()}, bag_size: {data.size(0)}')
        
        error = calculate_error(Y_hat, lbl)
        train_error += error
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
    
    # Calculate epoch statistics
    train_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
    
    if verbose:
        print(f'\nEpoch {epoch}: train_loss={train_loss:.4f}, '
              f'train_clustering_loss={train_inst_loss:.4f}, train_error={train_error:.4f}')
        for i in range(n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            if acc is not None:
                print(f'  Class {i}: acc={acc:.4f}, correct={correct}/{count}')
    
    return train_loss, train_error, train_inst_loss


def validate_clam(epoch, model, loader, n_classes, loss_fn, device, verbose=True):
    """
    Validation loop for CLAM.
    
    Returns:
        val_loss, val_error, val_auc, val_inst_loss
    """
    model.eval()
    acc_logger = AccuracyLogger(n_classes=n_classes)
    inst_logger = AccuracyLogger(n_classes=2)
    
    val_loss = 0.0
    val_error = 0.0
    val_inst_loss = 0.0
    inst_count = 0
    
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
            
            # Forward pass
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=lbl, instance_eval=True)
            
            acc_logger.log(Y_hat, lbl)
            
            # Losses
            loss = loss_fn(logits, lbl)
            val_loss += loss.item()
            
            instance_loss = instance_dict['instance_loss']
            inst_count += 1
            val_inst_loss += instance_loss.item()
            
            # Log instance predictions
            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            for p, t in zip(inst_preds, inst_labels):
                inst_logger.log(torch.tensor([p]), torch.tensor([t]))
            
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = lbl.item()
            
            error = calculate_error(Y_hat, lbl)
            val_error += error
    
    val_error /= len(loader)
    val_loss /= len(loader)
    
    if inst_count > 0:
        val_inst_loss /= inst_count
    
    # Calculate AUC
    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    else:
        from sklearn.preprocessing import label_binarize
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        auc = roc_auc_score(binary_labels, prob, multi_class='ovr', average='macro')
    
    if verbose:
        print(f'\nVal Set: val_loss={val_loss:.4f}, val_error={val_error:.4f}, auc={auc:.4f}')
        if inst_count > 0:
            for i in range(2):
                acc, correct, count = inst_logger.get_summary(i)
                if acc is not None:
                    print(f'  Instance class {i} clustering acc: {acc:.4f}, correct={correct}/{count}')
    
    return val_loss, val_error, auc, val_inst_loss


def main():
    parser = argparse.ArgumentParser(description='Train CLAM model')
    
    # Data paths
    parser.add_argument('--clinical_csv', type=str, default='clinical_data.csv',
                        help='Path to clinical data CSV')
    parser.add_argument('--features_dir', type=str, default='features',
                        help='Directory containing extracted features')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb'], 
                        default='clam_sb', help='Model type')
    parser.add_argument('--model_size', type=str, choices=['small', 'big'],
                        default='small', help='Model size')
    parser.add_argument('--embed_dim', type=int, default=2048,
                        help='Feature embedding dimension (2048 for ResNet50)')
    parser.add_argument('--n_classes', type=int, default=2,
                        help='Number of classes')
    parser.add_argument('--dropout', type=float, default=0.25,
                        help='Dropout rate')
    parser.add_argument('--k_sample', type=int, default=8,
                        help='Number of positive/negative patches to sample for clustering')
    
    # Training parameters
    parser.add_argument('--max_epochs', type=int, default=200,
                        help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--bag_weight', type=float, default=0.3,
                        help='Weight for bag loss (clustering weight = 1 - bag_weight)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    
    # Loss functions
    parser.add_argument('--bag_loss', type=str, choices=['ce', 'svm'], default='ce',
                        help='Bag-level loss function')
    parser.add_argument('--inst_loss', type=str, choices=['ce', 'svm'], default='svm',
                        help='Instance-level loss function')
    
    # Data split
    parser.add_argument('--test_size', type=float, default=0.15,
                        help='Test set proportion')
    parser.add_argument('--val_size', type=float, default=0.15,
                        help='Validation set proportion')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--create_new_splits', action='store_true',
                        help='Create new train/val/test splits')
    
    # Early stopping
    parser.add_argument('--early_stopping', action='store_true', default=True,
                        help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    
    # Other
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
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
    
    # Save arguments
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Print training configuration
    print("\n" + "="*60)
    print("CLAM Training Configuration")
    print("="*60)
    print(f"Model type: {args.model_type}")
    print(f"Learning rate: {args.lr}")
    print(f"Bag weight: {args.bag_weight} (Clustering weight: {1-args.bag_weight})")
    print(f"K sample: {args.k_sample}")
    print(f"Instance loss: {args.inst_loss}")
    print("="*60 + "\n")
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.clinical_csv,
        args.features_dir,
        batch_size=1,
        num_workers=args.num_workers,
        create_new_splits=args.create_new_splits,
        test_size=args.test_size,
        val_size=args.val_size,
        random_seed=args.seed
    )
    
    # Initialize loss functions
    print("\nInitializing loss functions...")
    if args.bag_loss == 'svm':
        bag_loss_fn = SmoothTop1SVM(n_classes=args.n_classes)
    else:
        bag_loss_fn = nn.CrossEntropyLoss()
    
    if args.inst_loss == 'svm':
        instance_loss_fn = SmoothTop1SVM(n_classes=2)
    else:
        instance_loss_fn = nn.CrossEntropyLoss()
    
    bag_loss_fn = bag_loss_fn.to(device)
    instance_loss_fn = instance_loss_fn.to(device)
    
    # Initialize model
    print("\nInitializing model...")
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
    
    # Initialize weights
    model.apply(initialize_weights)
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Initialize early stopping
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=args.patience, stop_epoch=50, verbose=True)
    else:
        early_stopping = None
    
    # Training loop
    print("\nStarting training...")
    best_auc = 0.0
    model_path = os.path.join(args.output_dir, 'model.pth')
    
    training_history = {
        'train_loss': [],
        'train_error': [],
        'train_inst_loss': [],
        'val_loss': [],
        'val_error': [],
        'val_auc': [],
        'val_inst_loss': []
    }
    
    for epoch in range(args.max_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.max_epochs}")
        print('='*60)
        
        # Training
        train_loss, train_error, train_inst_loss = train_loop_clam(
            epoch, model, train_loader, optimizer, args.n_classes,
            args.bag_weight, bag_loss_fn, device
        )
        
        # Validation
        val_loss, val_error, val_auc, val_inst_loss = validate_clam(
            epoch, model, val_loader, args.n_classes, bag_loss_fn, device
        )
        
        # Log history
        training_history['train_loss'].append(train_loss)
        training_history['train_error'].append(train_error)
        training_history['train_inst_loss'].append(train_inst_loss)
        training_history['val_loss'].append(val_loss)
        training_history['val_error'].append(val_error)
        training_history['val_auc'].append(val_auc)
        training_history['val_inst_loss'].append(val_inst_loss)
        
        # Save best model and check early stopping
        if early_stopping:
            stop = early_stopping(epoch, val_auc, model, model_path)
            if stop:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        else:
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), model_path)
                print(f"Saved best model with AUC: {best_auc:.4f}")
    
    # Save training history
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("Final Evaluation on Test Set")
    print("="*60)
    
    # Load best model
    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    test_loss, test_error, test_auc, test_inst_loss = validate_clam(
        0, model, test_loader, args.n_classes, bag_loss_fn, device
    )
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Error: {test_error:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  Accuracy: {1 - test_error:.4f}")
    
    # Save final results
    final_results = {
        'test_loss': test_loss,
        'test_error': test_error,
        'test_auc': test_auc,
        'test_accuracy': 1 - test_error,
        'best_val_auc': early_stopping.best_score if early_stopping else best_auc
    }
    
    with open(os.path.join(args.output_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nTraining complete. Model saved to {model_path}")
    print(f"Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
