"""
Inference script for CLAM model.
Generates attention maps on test set and ROC curves for train and test sets.

Usage:
    python infer.py --results_dir results_2026-01-30
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import pandas as pd
from PIL import Image, PngImagePlugin
import matplotlib.pyplot as plt

# Increase limit for large PNG images
PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024  # 100MB

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import CLAM_SB, CLAM_MB, SmoothTop1SVM
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


def create_model(config, embed_dim, device):
    """Create model from config."""
    instance_loss_fn = SmoothTop1SVM(n_classes=2).to(device)
    
    model_dict = {
        'gate': True,
        'size_arg': config.get('model_size', 'small'),
        'dropout': config.get('dropout', 0.5),
        'k_sample': config.get('k_sample', 8),
        'n_classes': 2,
        'instance_loss_fn': instance_loss_fn,
        'subtyping': False,
        'embed_dim': embed_dim
    }
    
    model_type = config.get('model_type', 'clam_sb')
    model = CLAM_SB(**model_dict) if model_type == 'clam_sb' else CLAM_MB(**model_dict)
    
    return model.to(device)


def get_predictions(model, loader, device):
    """Get predictions and labels for a dataset."""
    model.eval()
    probs_list = []
    labels_list = []
    patient_ids = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Getting predictions"):
            features = batch['features']
            label = batch['label']
            patient_id = batch['patient_id']
            
            if isinstance(features, list):
                features = features[0]
                label = label[0:1]
                patient_id = patient_id[0]
            
            data = features.to(device)
            
            _, Y_prob, _, _, _ = model(data, instance_eval=False)
            probs_list.append(Y_prob.cpu().numpy())
            labels_list.append(label.item())
            patient_ids.append(patient_id)
    
    probs = np.vstack(probs_list)
    labels = np.array(labels_list)
    
    return probs, labels, patient_ids


def get_attention_scores(model, features, device):
    """
    Extract attention scores from model.
    
    Args:
        model: CLAM model
        features: Input features (N, embed_dim)
        device: Computation device
        
    Returns:
        attention: Normalized attention scores (N,)
        Y_prob: Class probabilities
        Y_hat: Predicted class
    """
    features = features.to(device)
    
    with torch.no_grad():
        logits, Y_prob, Y_hat, A, _ = model(features, instance_eval=False)
        
        # A shape: (1, n_classes, N) for CLAM_SB or similar
        # We take the attention for the predicted class
        if len(A.shape) == 3:
            A = A.squeeze(0)  # (n_classes, N)
            attention = A[Y_hat.item()].cpu().numpy()  # (N,)
        else:
            attention = A.squeeze().cpu().numpy()
        
        # Min-max normalize attention
        min_val = attention.min()
        max_val = attention.max()
        if max_val - min_val > 0:
            attention = (attention - min_val) / (max_val - min_val)
        else:
            attention = np.zeros_like(attention)
    
    return attention, Y_prob.cpu().numpy(), Y_hat.item()


def create_attention_overlay(patient_id, attention_scores, coords, 
                             dataset_dir='dataset', output_path=None, 
                             tile_size=256, output_size=(1024, 1024), 
                             alpha=0.5, predicted_label=None, true_label=None):
    """
    Create attention overlay on the original image.
    High attention -> Red, Low attention -> Blue.
    
    Args:
        patient_id: Patient ID (e.g., '13901')
        attention_scores: Normalized attention scores (N,) in [0, 1]
        coords: Tile coordinates (N, 2) in (x, y) pixel format
        dataset_dir: Directory containing original images
        output_path: Path to save the overlay image
        tile_size: Size of each tile in the original image
        output_size: Output image size (width, height)
        alpha: Transparency of the attention overlay
        predicted_label: Predicted class label string
        true_label: Ground truth class label string
        
    Returns:
        PIL Image with attention overlay, or None if image not found
    """
    # Ensure patient_id is a string (fix for list bug)
    if isinstance(patient_id, list):
        patient_id = patient_id[0]
    patient_id = str(patient_id)
    
    # Try to find the original image
    image_path = None
    for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
        candidate = os.path.join(dataset_dir, f"{patient_id}{ext}")
        if os.path.exists(candidate):
            image_path = candidate
            break
    
    if image_path is None:
        print(f"Warning: Image not found for patient {patient_id} in {dataset_dir}")
        return None
    
    # Load original image
    original_img = Image.open(image_path).convert('RGBA')
    original_size = original_img.size  # (width, height)
    
    # Create attention overlay at original resolution
    attention_overlay = Image.new('RGBA', original_size, (0, 0, 0, 0))
    
    # Create blue-to-red colormap for attention
    # Low attention (0) -> Blue, High attention (1) -> Red
    for i, (coord, attn) in enumerate(zip(coords, attention_scores)):
        x, y = int(coord[0]), int(coord[1])
        
        # Interpolate color: Blue (low) -> Red (high)
        red = int(255 * attn)
        blue = int(255 * (1 - attn))
        green = 0
        alpha_channel = int(255 * alpha)
        
        # Create colored tile
        tile_color = Image.new('RGBA', (tile_size, tile_size), (red, green, blue, alpha_channel))
        
        # Paste at the correct position
        attention_overlay.paste(tile_color, (x, y))
    
    # Composite overlay on original image
    composite = Image.alpha_composite(original_img, attention_overlay)
    
    # Resize to output size
    composite_resized = composite.resize(output_size, Image.Resampling.LANCZOS)
    
    # Convert to RGB for saving
    final_img = composite_resized.convert('RGB')
    
    # Create figure with title
    fig, ax = plt.subplots(figsize=(10, 10), dpi=102.4)  # 10*102.4 ≈ 1024
    ax.imshow(final_img)
    ax.axis('off')
    
    # Add title with predicted and true labels
    pred_str = predicted_label if predicted_label else '?'
    true_str = true_label if true_label else '?'
    
    # Color code: green if correct, red if incorrect
    if predicted_label and true_label:
        # Extract status from label strings for comparison
        pred_correct = predicted_label.split('status')[1].strip().rstrip(')') if 'status' in predicted_label else None
        true_correct = true_label.split('status')[1].strip().rstrip(')') if 'status' in true_label else None
        title_color = 'green' if pred_correct == true_correct else 'red'
    else:
        title_color = 'black'
    
    title = f"Patient {patient_id}\nPrédiction: {pred_str} | Vérité terrain: {true_str}"
    ax.set_title(title, fontsize=14, fontweight='bold', color=title_color)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=102.4, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.close()
    
    return final_img


def generate_attention_maps(model, loader, device, output_dir, dataset_dir='dataset', 
                            tile_size=256, output_size=(1024, 1024)):
    """
    Generate attention maps for all patients in a dataset.
    
    Args:
        model: Trained CLAM model
        loader: DataLoader for dataset
        device: Computation device
        output_dir: Directory to save attention maps
        dataset_dir: Directory containing original images
        tile_size: Size of tiles
        output_size: Output image size
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    class_names = {0: 'responder', 1: 'progressor'}
    results = []
    
    print(f"\nGenerating attention maps for {len(loader)} patients...")
    
    for batch in tqdm(loader, desc="Generating attention maps"):
        features = batch['features']
        label = batch['label']
        patient_id = batch['patient_id']
        coords = batch['coords']
        
        if isinstance(features, list):
            features = features[0]
            label = label[0:1]
            patient_id = patient_id[0]
            coords = coords[0]
        
        # Ensure patient_id is a string
        if isinstance(patient_id, list):
            patient_id = patient_id[0]
        patient_id = str(patient_id)
        
        # Get attention scores
        data = features.to(device)
        attention, Y_prob, Y_hat = get_attention_scores(model, data, device)
        
        true_label = label.item()
        pred_label = Y_hat
        
        # Create filename: patient_id_prediction_status
        pred_name = class_names[pred_label]
        true_name = class_names[true_label]
        filename = f"{patient_id}_{pred_name}_pred_status{pred_label}_{true_name}_true_status{true_label}.png"
        output_path = os.path.join(output_dir, filename)
        
        # Get coordinates as numpy array
        if isinstance(coords, torch.Tensor):
            coords_np = coords.numpy()
        else:
            coords_np = np.array(coords)
        
        # Create attention overlay
        create_attention_overlay(
            patient_id=patient_id,
            attention_scores=attention,
            coords=coords_np,
            dataset_dir=dataset_dir,
            output_path=output_path,
            tile_size=tile_size,
            output_size=output_size,
            alpha=0.5,
            predicted_label=f"{pred_name} (status {pred_label})",
            true_label=f"{true_name} (status {true_label})"
        )
        
        # Store results
        results.append({
            'patient_id': patient_id,
            'true_label': int(true_label),
            'predicted_label': int(pred_label),
            'probability_progressor': float(Y_prob[0, 1]),
            'correct': true_label == pred_label
        })
    
    return results


def plot_roc_curves(train_probs, train_labels, test_probs, test_labels, output_path):
    """
    Plot ROC curves for train and test sets.
    
    Args:
        train_probs: Predicted probabilities for train set (N, 2)
        train_labels: True labels for train set
        test_probs: Predicted probabilities for test set (N, 2)
        test_labels: True labels for test set
        output_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Colors
    train_color = '#2196F3'  # Blue
    test_color = '#F44336'   # Red
    
    # ----- Plot 1: Train ROC -----
    fpr_train, tpr_train, _ = roc_curve(train_labels, train_probs[:, 1])
    roc_auc_train = auc(fpr_train, tpr_train)
    
    axes[0].plot(fpr_train, tpr_train, color=train_color, lw=2, 
                 label=f'Train AUC = {roc_auc_train:.3f}')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance')
    axes[0].fill_between(fpr_train, tpr_train, alpha=0.2, color=train_color)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('Taux de Faux Positifs (1 - Spécificité)', fontsize=11)
    axes[0].set_ylabel('Taux de Vrais Positifs (Sensibilité)', fontsize=11)
    axes[0].set_title('Courbe ROC - Ensemble d\'Entraînement', fontsize=12, fontweight='bold')
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')
    
    # ----- Plot 2: Test ROC -----
    fpr_test, tpr_test, _ = roc_curve(test_labels, test_probs[:, 1])
    roc_auc_test = auc(fpr_test, tpr_test)
    
    axes[1].plot(fpr_test, tpr_test, color=test_color, lw=2, 
                 label=f'Test AUC = {roc_auc_test:.3f}')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance')
    axes[1].fill_between(fpr_test, tpr_test, alpha=0.2, color=test_color)
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Taux de Faux Positifs (1 - Spécificité)', fontsize=11)
    axes[1].set_ylabel('Taux de Vrais Positifs (Sensibilité)', fontsize=11)
    axes[1].set_title('Courbe ROC - Ensemble de Test', fontsize=12, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')
    
    # ----- Plot 3: Combined ROC -----
    axes[2].plot(fpr_train, tpr_train, color=train_color, lw=2, 
                 label=f'Train AUC = {roc_auc_train:.3f}')
    axes[2].plot(fpr_test, tpr_test, color=test_color, lw=2, 
                 label=f'Test AUC = {roc_auc_test:.3f}')
    axes[2].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance')
    axes[2].set_xlim([0.0, 1.0])
    axes[2].set_ylim([0.0, 1.05])
    axes[2].set_xlabel('Taux de Faux Positifs (1 - Spécificité)', fontsize=11)
    axes[2].set_ylabel('Taux de Vrais Positifs (Sensibilité)', fontsize=11)
    axes[2].set_title('Comparaison Train vs Test', fontsize=12, fontweight='bold')
    axes[2].legend(loc='lower right', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curves saved to {output_path}")
    print(f"  Train AUC: {roc_auc_train:.4f}")
    print(f"  Test AUC: {roc_auc_test:.4f}")
    
    return roc_auc_train, roc_auc_test


def main():
    parser = argparse.ArgumentParser(description='CLAM Inference and Visualization')
    
    # Required: results directory
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to results directory (e.g., results_2026-01-30)')
    
    # Data paths
    parser.add_argument('--clinical_csv', type=str, default='clinical_data.csv')
    parser.add_argument('--features_dir', type=str, default='features')
    parser.add_argument('--dataset_dir', type=str, default='dataset',
                        help='Directory containing original images for attention maps')
    
    # Model settings
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Attention maps settings
    parser.add_argument('--tile_size', type=int, default=256,
                        help='Tile size in pixels')
    parser.add_argument('--output_size', type=int, default=1024,
                        help='Output image size for attention maps')
    
    # Options
    parser.add_argument('--skip_attention_maps', action='store_true',
                        help='Skip attention map generation')
    parser.add_argument('--skip_roc', action='store_true',
                        help='Skip ROC curve generation')
    
    args = parser.parse_args()
    
    # Check results directory exists
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    # Check for required files
    model_path = os.path.join(args.results_dir, 'best_model.pt')
    config_path = os.path.join(args.results_dir, 'best_config.json')
    splits_path = os.path.join(args.results_dir, 'splits.json')
    
    for path, name in [(model_path, 'best_model.pt'), 
                        (config_path, 'best_config.json'),
                        (splits_path, 'splits.json')]:
        if not os.path.exists(path):
            print(f"Error: Required file not found: {path}")
            sys.exit(1)
    
    # Setup device
    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"\nLoaded config from {config_path}")
    print(f"  bag_weight: {config.get('bag_weight')}")
    print(f"  dropout: {config.get('dropout')}")
    print(f"  k_sample: {config.get('k_sample')}")
    
    # Load splits
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    train_patients = np.array(splits['trainval'])
    test_patients = np.array(splits['test'])
    print(f"\nLoaded splits from {splits_path}")
    print(f"  Train patients: {len(train_patients)}")
    print(f"  Test patients: {len(test_patients)}")
    
    # Load data
    print("\nLoading data...")
    _, _, df = get_patients_and_labels(args.clinical_csv, args.features_dir)
    
    # Create datasets
    train_dataset = CLAMDataset(train_patients, df, args.features_dir)
    test_dataset = CLAMDataset(test_patients, df, args.features_dir)
    
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    
    # Create and load model
    print("\nLoading model...")
    model = create_model(config, args.embed_dim, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")
    
    # =========================================================================
    # GENERATE ROC CURVES
    # =========================================================================
    if not args.skip_roc:
        print(f"\n{'='*60}")
        print("GENERATING ROC CURVES")
        print(f"{'='*60}")
        
        # Get predictions
        print("\nGetting train predictions...")
        train_probs, train_labels, _ = get_predictions(model, train_loader, device)
        
        print("Getting test predictions...")
        test_probs, test_labels, _ = get_predictions(model, test_loader, device)
        
        # Plot ROC curves
        roc_path = os.path.join(args.results_dir, 'roc_curves.png')
        train_auc, test_auc = plot_roc_curves(
            train_probs, train_labels,
            test_probs, test_labels,
            roc_path
        )
        
        # Calculate and print additional metrics
        print(f"\n--- Train Set Metrics ---")
        train_preds = np.argmax(train_probs, axis=1)
        train_acc = accuracy_score(train_labels, train_preds)
        train_cm = confusion_matrix(train_labels, train_preds)
        print(f"Accuracy: {train_acc:.4f}")
        print(f"Confusion Matrix:\n{train_cm}")
        
        print(f"\n--- Test Set Metrics ---")
        test_preds = np.argmax(test_probs, axis=1)
        test_acc = accuracy_score(test_labels, test_preds)
        test_cm = confusion_matrix(test_labels, test_preds)
        print(f"Accuracy: {test_acc:.4f}")
        print(f"Confusion Matrix:\n{test_cm}")
    
    # =========================================================================
    # GENERATE ATTENTION MAPS
    # =========================================================================
    if not args.skip_attention_maps:
        print(f"\n{'='*60}")
        print("GENERATING ATTENTION MAPS")
        print(f"{'='*60}")
        
        attention_maps_dir = os.path.join(args.results_dir, 'attention_maps')
        attention_results = generate_attention_maps(
            model, test_loader, device, attention_maps_dir,
            dataset_dir=args.dataset_dir,
            tile_size=args.tile_size,
            output_size=(args.output_size, args.output_size)
        )
        
        # Calculate and print summary
        correct = sum(1 for r in attention_results if r['correct'])
        total = len(attention_results)
        print(f"\nAttention maps generated: {total}")
        print(f"Correct predictions: {correct}/{total} ({100*correct/total:.1f}%)")
        
        # Save attention results
        attention_results_path = os.path.join(args.results_dir, 'attention_results.json')
        with open(attention_results_path, 'w') as f:
            json.dump(attention_results, f, indent=2)
        print(f"Attention results saved to {attention_results_path}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*60}")
    print("INFERENCE COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {args.results_dir}/")
    if not args.skip_roc:
        print(f"  - roc_curves.png: Train and Test ROC curves")
    if not args.skip_attention_maps:
        print(f"  - attention_maps/: Attention overlays on test images")
        print(f"  - attention_results.json: Detailed prediction results")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
